from __future__ import annotations

import torch
from reward_service.schemas import ScoreRequest, ScoreResponse

from unirl.reward.remote import RemoteRewardBackend, RemoteRewardSpec
from unirl.types.primitives import Images, Texts
from unirl.types.reward import RewardRequest


def test_remote_image_payload_matches_service_schema() -> None:
    backend = RemoteRewardBackend(
        config=RemoteRewardSpec(
            base_url="http://unused",
            required_rewards=("pickscore",),
            expected_scorer_version="1",
        ),
        base_device="cpu",
    )
    request = RewardRequest(
        primitives={"text": Texts(texts=["prompt"])},
        generated={"image": Images(pixels=torch.rand(1, 3, 8, 8))},
        sample_ids=["sample-0"],
        group_ids=["group-0"],
    )

    wire = ScoreRequest.model_validate(backend._build_score_payload(request))
    assert wire.protocol_version == "1"
    assert wire.requests[0].sample_id == "sample-0"
    assert wire.requests[0].group_id == "group-0"
    assert wire.requests[0].idempotency_key
    backend.dispose()


def test_legacy_score_response_remains_schema_compatible() -> None:
    response = ScoreResponse.model_validate(
        {
            "results": [{"pickscore": {"score": 1.0}}],
            "errors": [{}],
        }
    )
    assert response.protocol_version == "1"
    assert response.identities == []
