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
        {
            "index": i,
            "relevance_score": float(scores[i]),  # changed from "score"
            "document": req.documents[i]
        }
        for i in range(len(req.documents))
    ]
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return {"results": results[:req.top_n]}


@register(
    "astrbot_plugin_rerank_lmstudio",
    "MKiyoaki",
    "Provide a local /v1/rerank endpoint powered by sentence-transformers",
    "0.1.2",
    "https://github.com/MKiyoaki/astrbot_plugin_rerank_lmstudio"
)
class RerankPlugin(Star):
    def __init__(self, context, config: AstrBotConfig):
        """Initialize model and start FastAPI server on port 8001."""
        super().__init__(context)
        self.server = None  # initialize first to avoid AttributeError in terminate
        global model

        model_name = config.get(
            "model_name", "BAAI/bge-reranker-base")  # smaller default

        logger.info(f"[Rerank] Loading model: {model_name}...")
        model = CrossEncoder(model_name)
        logger.info("[Rerank] Model loaded.")

        cfg = uvicorn.Config(app, host="0.0.0.0", port=8001)
        self.server = uvicorn.Server(cfg)

        self.server_thread = threading.Thread(
            target=self.server.run,
            daemon=True
        )
        self.server_thread.start()
        logger.info("[Rerank] Service started on port 8001")

    async def terminate(self):
        """Gracefully shut down the uvicorn server."""
        logger.info("[Rerank] Shutting down service...")
        if self.server is not None:
            self.server.should_exit = True
            self.server_thread.join(timeout=5)
        logger.info("[Rerank] Service stopped.")
