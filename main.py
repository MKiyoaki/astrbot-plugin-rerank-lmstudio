import os
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from astrbot.api import logger, AstrBotConfig
from astrbot.api.star import Star, register
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

PLUGIN_NAME = "astrbot_plugin_rerank_lmstudio"
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"

app = FastAPI()
model = None


class RerankRequest(BaseModel):
    """Request schema for /v1/rerank endpoint."""

    query: str
    documents: list[str]
    top_n: int = 5


@app.post("/v1/rerank")
def rerank(req: RerankRequest):
    """Score and rank documents by relevance to the query."""
    pairs = [[req.query, doc] for doc in req.documents]
    scores = model.predict(pairs)
    results = [
        {
            "index": i,
            "relevance_score": float(scores[i]),
            "document": req.documents[i],
        }
        for i in range(len(req.documents))
    ]
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return {"results": results[: req.top_n]}


def _resolve_plugin_data_dir() -> Path:
    """Return the plugin data directory under AstrBot's data root.

    Per the AstrBot plugin storage guide, large files belong under
    ``data/plugin_data/{plugin_name}/`` rather than the plugin source
    directory, so they survive plugin reinstall or update.
    """
    base = Path(get_astrbot_data_path()) / "plugin_data" / PLUGIN_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base


def _attempt_snapshot_download(
    model_id: str,
    target_dir: Path,
    endpoint: str | None,
) -> None:
    """Run a single snapshot_download attempt against the given endpoint.

    Setting ``HF_ENDPOINT`` must happen before ``huggingface_hub`` reads it,
    so the import is performed lazily inside this function.
    """
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
        logger.info(f"[Rerank] Using HF endpoint: {endpoint}")
    else:
        os.environ.pop("HF_ENDPOINT", None)

    from huggingface_hub import snapshot_download

    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.tflite"],
    )


def _ensure_model_available(model_id: str, cache_root: Path) -> Path:
    """Guarantee a complete local snapshot, downloading once if required.

    On the first run, the model is fetched from huggingface.co; if that
    fails for any reason (network reset, DNS issue, region block), the
    function retries against ``hf-mirror.com`` before giving up. A sentinel
    file marks completion so subsequent boots skip the network entirely.
    """
    target_dir = cache_root / "models" / model_id.replace("/", "__")
    sentinel = target_dir / ".download_complete"

    if sentinel.exists():
        logger.info(f"[Rerank] Reusing cached model at {target_dir}")
        return target_dir

    logger.info(
        f"[Rerank] Model not cached locally; downloading '{model_id}' "
        f"to {target_dir} (one-time)."
    )

    last_error: Exception | None = None
    attempts: list[tuple[str, str | None]] = [
        ("huggingface.co", None),
        ("hf-mirror.com", HF_MIRROR_ENDPOINT),
    ]

    for label, endpoint in attempts:
        try:
            logger.info(f"[Rerank] Attempting download via {label} ...")
            _attempt_snapshot_download(model_id, target_dir, endpoint)
            sentinel.touch()
            logger.info(f"[Rerank] Model download finished via {label}.")
            return target_dir
        except Exception as exc:  # noqa: BLE001 - broad net required for retry
            last_error = exc
            logger.warning(
                f"[Rerank] Download via {label} failed: {exc!r}. "
                f"Trying next source if available."
            )

    raise RuntimeError(
        f"[Rerank] All download attempts for '{model_id}' failed. "
        f"Last error: {last_error!r}"
    )


@register(
    "astrbot_plugin_rerank_lmstudio",
    "MKiyoaki",
    "Provide a local /v1/rerank endpoint powered by sentence-transformers",
    "0.1.3",
    "https://github.com/MKiyoaki/astrbot_plugin_rerank_lmstudio",
)
class RerankPlugin(Star):
    def __init__(self, context, config: AstrBotConfig):
        """Resolve the local model, then start the FastAPI server."""
        super().__init__(context)
        self.server = None
        self.server_thread = None
        global model

        model_id = config.get("model_name", "BAAI/bge-reranker-base")
        custom_model_path = (config.get("model_path", "") or "").strip()
        port = int(config.get("port", 8001))

        plugin_data_dir = _resolve_plugin_data_dir()
        logger.info(f"[Rerank] Plugin data directory: {plugin_data_dir}")

        if custom_model_path:
            model_dir = Path(custom_model_path).expanduser().resolve()
            if not model_dir.exists():
                raise FileNotFoundError(
                    f"[Rerank] Configured model_path does not exist: {model_dir}"
                )
            logger.info(f"[Rerank] Loading user-specified model: {model_dir}")
        else:
            model_dir = _ensure_model_available(
                model_id=model_id,
                cache_root=plugin_data_dir,
            )

        # Snapshot is on disk; force offline so neither huggingface_hub nor
        # transformers issues a remote ETag check on subsequent loads.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        # Imported after the offline flags are set, so they are honoured.
        from sentence_transformers import CrossEncoder

        logger.info(f"[Rerank] Loading CrossEncoder from {model_dir}")
        try:
            model = CrossEncoder(str(model_dir), local_files_only=True)
        except TypeError:
            # Older sentence-transformers builds lack the kwarg; the env
            # vars above are sufficient on those versions.
            model = CrossEncoder(str(model_dir))
        logger.info("[Rerank] Model loaded.")

        cfg = uvicorn.Config(app, host="0.0.0.0", port=port)
        self.server = uvicorn.Server(cfg)
        self.server_thread = threading.Thread(
            target=self.server.run, daemon=True)
        self.server_thread.start()
        logger.info(f"[Rerank] Service started on port {port}")

    async def terminate(self):
        """Gracefully shut down the uvicorn server."""
        logger.info("[Rerank] Shutting down service...")
        if self.server is not None:
            self.server.should_exit = True
        if self.server_thread is not None:
            self.server_thread.join(timeout=5)
        logger.info("[Rerank] Service stopped.")
