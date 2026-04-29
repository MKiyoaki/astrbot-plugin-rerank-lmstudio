from astrbot.api.star import Star, register
from astrbot.api import logger, AstrBotConfig
import threading
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

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
        {"index": i, "score": float(scores[i]), "document": req.documents[i]}
        for i in range(len(req.documents))
    ]
    results.sort(key=lambda x: x["score"], reverse=True)
    return {"results": results[:req.top_n]}


@register(
    "astrbot_plugin_rerank_lmstudio",
    "MKiyoaki",
    "Provide a local /v1/rerank endpoint powered by sentence-transformers",
    "0.1.0",
    "https://github.com/MKiyoaki/astrbot_plugin_rerank_lmstudio"
)
class RerankPlugin(Star):
    def __init__(self, context, config: AstrBotConfig):
        """Initialize model and start FastAPI server in background thread."""
        super().__init__(context)  # only context, not config
        global model

        model_name = config.get("model_name", "BAAI/bge-reranker-v2-m3")
        port = config.get("port", 8001)

        logger.info(f"[Rerank] Loading model: {model_name}...")
        model = CrossEncoder(model_name)
        logger.info("[Rerank] Model loaded.")

        self.server_thread = threading.Thread(
            target=lambda: uvicorn.run(app, host="0.0.0.0", port=port),
            daemon=True
        )
        self.server_thread.start()
        logger.info(f"[Rerank] Service started on port {port}")

    async def terminate(self):
        """Called when plugin is unloaded."""
        logger.info("[Rerank] Plugin unloaded, service stopped.")
        