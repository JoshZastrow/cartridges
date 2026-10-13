"""Model registry: resolve, verify, and promote OCR adapters.

The registry is the HF Hub repo that training pushes to (TrainConfig.hf_repo_id).
- resolve_revision(): a tag or branch -> the commit SHA. "base" stays "base"
  (the base model, no adapter).
- check_promotion(): served-model eval metrics -> pass or fail, with the reasons.
- tag_verified(): adds the tag verified-<date>-<sha7> and moves the tag prod.

The record of the deployed model is in a Modal Dict, so a client can read
the served SHA without a request to the GPU server (no cold start).
"""

from __future__ import annotations

import datetime

from pydantic import BaseModel

from cartridges.serve.recipe import OcrRecipe
from cartridges.training.config import TrainConfig

BASE = "base"
HF_REPO_ID = TrainConfig().hf_repo_id
PROD_TAG = "prod"
DEPLOY_DICT = "cartridges-ocr-deploy"


def resolve_revision(revision: str, repo_id: str = HF_REPO_ID) -> str:
    if revision == BASE:
        return BASE
    from huggingface_hub import HfApi

    return HfApi().model_info(repo_id, revision=revision).sha


class PromotionResult(BaseModel):
    passed: bool
    reasons: list[str]


def check_promotion(
    *,
    n: int,
    n_errors: int,
    corpus_cer: float | None,
    loop_rate: float,
    max_cer: float,
    max_loop_rate: float,
    max_error_rate: float = 0.0,
) -> PromotionResult:
    """Check the metrics of a served-model eval (serve.evaluate.score)
    against the thresholds. corpus_cer is the headline CER."""
    reasons = []
    if n == 0:
        reasons.append("no predictions")
    if corpus_cer is None or corpus_cer > max_cer:
        reasons.append(f"corpus_cer {corpus_cer} > {max_cer}")
    if loop_rate > max_loop_rate:
        reasons.append(f"loop_rate {loop_rate:.4f} > {max_loop_rate}")
    if n and n_errors / n > max_error_rate:
        reasons.append(f"error rate {n_errors}/{n} > {max_error_rate}")
    return PromotionResult(passed=not reasons, reasons=reasons)


def tag_verified(sha: str, repo_id: str = HF_REPO_ID, today: datetime.date | None = None) -> list[str]:
    """Add verified-<date>-<sha7> and move prod to `sha`. Hub tags do not
    move, so delete prod first, then make it again."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RevisionNotFoundError

    api = HfApi()
    verified = f"verified-{(today or datetime.date.today()).isoformat()}-{sha[:7]}"
    api.create_tag(repo_id, tag=verified, revision=sha, exist_ok=True)
    try:
        api.delete_tag(repo_id, tag=PROD_TAG)
    except RevisionNotFoundError:
        pass
    api.create_tag(repo_id, tag=PROD_TAG, revision=sha)
    return [verified, PROD_TAG]


def record_deploy(revision: str, recipe: OcrRecipe) -> dict:
    import modal

    record = {
        "revision": revision,
        "model_sha": recipe.model_sha,
        "model_id": recipe.model_id,
        "recipe": recipe.model_dump(mode="json"),
        "deployed_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    }
    modal.Dict.from_name(DEPLOY_DICT, create_if_missing=True)["serve"] = record
    return record


def deployed() -> dict | None:
    """The record that record_deploy() wrote, or None."""
    import modal

    return modal.Dict.from_name(DEPLOY_DICT, create_if_missing=True).get("serve")
