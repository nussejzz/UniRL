"""Index schedulers used by GRPO-style algorithms."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import List, Literal, Optional, Set, Tuple, Union

import numpy as np

Strategy = Literal["all", "progressive", "random", "decay", "exp_decay"]


@dataclass
class WindowConfig:
    """Configuration for stateless window-based index scheduling."""

    strategy: Strategy = "all"
    window_size: int = 4
    iters_per_window: int = 25
    init_timestep: int = 0
    overlap_size: int = 0
    roll_back: bool = False
    max_iters_per_window: Optional[int] = None
    min_iters_per_window: Optional[int] = None
    exp_decay_threshold: int = 13
    exp_decay_k: float = 0.1

    def __post_init__(self) -> None:
        if self.strategy == "decay":
            if self.max_iters_per_window is None:
                self.max_iters_per_window = self.iters_per_window
            if self.min_iters_per_window is None:
                self.min_iters_per_window = max(1, self.iters_per_window // 4)


class TimestepScheduler(ABC):
    """Abstract base class for stateless timestep-index schedulers."""

    def __init__(self, num_timesteps: int):
        self.num_timesteps = num_timesteps

    @abstractmethod
    def get_sde_indices(self, step: Optional[int] = None) -> Set[int]:
        """Return the selected indices for the given step."""


def normalize_timestep_fraction(
    timestep_fraction: Union[float, Tuple[float, float], List[float]],
) -> Tuple[float, float]:
    """Normalize timestep_fraction to a ``(start, end)`` tuple."""
    if isinstance(timestep_fraction, Sequence):
        if len(timestep_fraction) != 2:
            raise ValueError(f"timestep_fraction tuple must have exactly 2 elements, got {len(timestep_fraction)}")
        start, end = float(timestep_fraction[0]), float(timestep_fraction[1])
    else:
        start, end = 0.0, float(timestep_fraction)
    if not (0.0 <= start <= 1.0) or not (0.0 <= end <= 1.0):
        raise ValueError(f"timestep_fraction values must be in [0.0, 1.0], got ({start}, {end})")
    if start > end:
        raise ValueError(f"timestep_fraction start ({start}) must be <= end ({end})")
    return (start, end)


class AllSDEScheduler(TimestepScheduler):
    """Full-range index scheduler with optional range filtering and sparse sampling."""

    def __init__(
        self,
        num_timesteps: int,
        timestep_fraction: Union[float, Tuple[float, float]] = 1.0,
        num_sde_steps: Optional[int] = None,
        seed: int = 0,
    ):
        super().__init__(num_timesteps)
        self.timestep_fraction = timestep_fraction
        self.num_sde_steps = num_sde_steps
        self.seed = int(seed)
        self._fraction_start, self._fraction_end = normalize_timestep_fraction(timestep_fraction)
        self._effective_start = int(num_timesteps * self._fraction_start)
        self._effective_end = int(num_timesteps * self._fraction_end)
        if num_sde_steps is not None:
            pool_size = self._effective_end - self._effective_start
            if num_sde_steps > pool_size:
                raise ValueError(
                    f"num_sde_steps ({num_sde_steps}) exceeds available timesteps "
                    f"in fraction range [{self._effective_start}, {self._effective_end}) "
                    f"(pool_size={pool_size})"
                )
            if num_sde_steps < 0:
                raise ValueError(f"num_sde_steps must be non-negative, got {num_sde_steps}")

    def get_sde_indices(self, step: Optional[int] = None) -> Set[int]:
        if self.num_sde_steps == 0:
            return set()
        pool = list(range(self._effective_start, self._effective_end))
        if self.num_sde_steps is None or self.num_sde_steps >= len(pool):
            return set(pool)
        seed = self.seed if step is None else self.seed + int(step)
        rng = np.random.default_rng(seed)
        chosen = rng.choice(pool, size=self.num_sde_steps, replace=False)
        return set(int(i) for i in chosen)


class RandomContiguousSDEScheduler(TimestepScheduler):
    """Select one reproducible contiguous SDE window per rollout."""

    def __init__(
        self,
        num_timesteps: int,
        window_size: int = 3,
        start: int = 0,
        end: Optional[int] = None,
        seed: int = 0,
        **_inherited: object,
    ):
        super().__init__(num_timesteps)
        self.window_size = int(window_size)
        self.start = int(start)
        self.end = int(num_timesteps if end is None else end)
        self.seed = int(seed)
        if self.start < 0 or self.end > int(num_timesteps) or self.start >= self.end:
            raise ValueError(
                f"Invalid contiguous SDE range [{self.start}, {self.end}) for num_timesteps={num_timesteps}"
            )
        if self.window_size <= 0 or self.window_size > self.end - self.start:
            raise ValueError(
                f"window_size={self.window_size} must be in [1, {self.end - self.start}] "
                f"for range [{self.start}, {self.end})"
            )

    def get_sde_indices(self, step: Optional[int] = None) -> Set[int]:
        rng = np.random.default_rng(self.seed if step is None else self.seed + int(step))
        window_start = int(rng.integers(self.start, self.end - self.window_size + 1))
        return set(range(window_start, window_start + self.window_size))


class SigmaBandSDEScheduler(TimestepScheduler):
    """Select one transition from each band on a shifted sigma schedule."""

    def __init__(
        self,
        num_timesteps: int,
        shift: float,
        sigma_bands: Sequence[Sequence[float]],
        seed: int = 0,
        timestep_fraction: object = None,
        num_sde_steps: object = None,
    ):
        super().__init__(num_timesteps)
        # Hydra deep-merges this scheduler over the parent AllSDEScheduler.
        # These inherited knobs have no meaning once explicit sigma bands own
        # the candidate pools.
        del timestep_fraction, num_sde_steps
        if int(num_timesteps) <= 0:
            raise ValueError(f"num_timesteps must be positive, got {num_timesteps}")
        self.shift = float(shift)
        if self.shift <= 0:
            raise ValueError(f"shift must be positive, got {shift}")
        self.seed = int(seed)
        self.sigma_bands = tuple((float(band[0]), float(band[1])) for band in sigma_bands)
        if not self.sigma_bands:
            raise ValueError("sigma_bands must be non-empty")

        sigmas = [self._sigma_at(index) for index in range(self.num_timesteps)]
        pools: list[tuple[int, ...]] = []
        used: set[int] = set()
        for band_index, (low, high) in enumerate(self.sigma_bands):
            if not (0.0 <= low < high <= 1.0):
                raise ValueError(f"sigma band {band_index} must satisfy 0 <= low < high <= 1, got {(low, high)}")
            pool = tuple(index for index, sigma in enumerate(sigmas) if low <= sigma <= high)
            if not pool:
                raise ValueError(
                    f"sigma band {band_index} {(low, high)} selects no transition "
                    f"for num_timesteps={self.num_timesteps}, shift={self.shift}"
                )
            overlap = used.intersection(pool)
            if overlap:
                raise ValueError(
                    f"sigma bands must select disjoint transition pools; band {band_index} overlaps at {sorted(overlap)}"
                )
            used.update(pool)
            pools.append(pool)
        self._pools = tuple(pools)

    def _sigma_at(self, index: int) -> float:
        t = 1.0 - float(index) / float(self.num_timesteps)
        return (self.shift * t) / (1.0 + (self.shift - 1.0) * t)

    @property
    def candidate_pools(self) -> tuple[tuple[int, ...], ...]:
        return self._pools

    def get_sde_indices(self, step: Optional[int] = None) -> Set[int]:
        seed = self.seed if step is None else self.seed + int(step)
        rng = np.random.default_rng(seed)
        return {int(rng.choice(pool)) for pool in self._pools}


class WindowScheduler(TimestepScheduler):
    """Stateless sliding-window index scheduler."""

    WINDOW_STRATEGY_TO_METHOD_NAME = {
        "all": None,
        "progressive": "_resolve_progressive",
        "random": "_resolve_random",
    }

    def __init__(self, num_timesteps: int, config: WindowConfig):
        super().__init__(num_timesteps)
        self.config = config
        if self.config.strategy not in self.WINDOW_STRATEGY_TO_METHOD_NAME:
            raise ValueError(
                f"Bad strategy configuration for WindowScheduler: {self.config.strategy}. "
                f"Available options: {set(self.WINDOW_STRATEGY_TO_METHOD_NAME.keys())}"
            )

    def get_sde_indices(self, step: Optional[int] = None) -> Set[int]:
        if self.config.strategy == "all":
            return set(range(self.num_timesteps))
        resolve_method = getattr(
            self,
            self.WINDOW_STRATEGY_TO_METHOD_NAME[self.config.strategy],
        )
        return resolve_method(0 if step is None else int(step))

    def _resolve_progressive(self, step: int) -> Set[int]:
        window_step = step // self.config.iters_per_window
        stride = self.config.window_size - self.config.overlap_size
        remaining = self.num_timesteps - self.config.init_timestep - self.config.window_size
        num_one_round_window_steps = max(1, remaining // stride + 1)
        if window_step >= num_one_round_window_steps and not self.config.roll_back:
            window_step = num_one_round_window_steps - 1
            return self._resolve_progressive(window_step * self.config.iters_per_window)

        window_step = window_step % num_one_round_window_steps
        cur_timestep = self.config.init_timestep + window_step * stride
        return set(range(cur_timestep, cur_timestep + self.config.window_size))

    def _resolve_random(self, step: int) -> Set[int]:
        rng = np.random.default_rng(step)
        max_start = max(0, self.num_timesteps - self.config.window_size)
        cur_timestep = int(rng.integers(0, max_start + 1))
        return set(range(cur_timestep, cur_timestep + self.config.window_size))
