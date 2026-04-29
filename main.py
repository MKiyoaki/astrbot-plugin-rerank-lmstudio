from astrbot.api.star import Star, register
from astrbot.api import logger
import threading
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

app = FastAPI()
model = None

class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    top_n: int = 5

@app.post("/v1/rerank")
def rerank(req: RerankRequest):
    pairs = [[req.query, doc] for doc in req.documents]
    scores = model.predict(pairs)
    results = [
        {"index": i, "score": float(scores[i]), "document": req.documents[i]}
        for i in range(len(req.documents))
    ]
    results.sort(key=lambda x: x["score"], reverse=True)
    return {"results": results[:req.top_n]}

@register("astrbot_plugin_rerank_lmstudio", "MKiyoaki", "Provide a local /v1/rerank endpoint powered by sentence-transformers", "0.1.0")
class RerankPlugin(Star):
    def __init__(self, context, config):
        super().__init__(context, config)
        global model
        
        model_name = config.get("model_name", "BAAI/bge-reranker-v2-m3")
        port = config.get("port", 8001)
        
        logger.info(f"[Rerank] Loading model: {model_name}...")
        model = CrossEncoder(model_name)
        logger.info("[Rerank] Model loading completed. ")
        
        self.server_thread = threading.Thread(
            target=lambda: uvicorn.run(app, host="0.0.0.0", port=port),
            daemon=True
        )
        self.server_thread.start()
        logger.info(f"[Rerank] Service begins. Port: {port}")

    async def terminate(self):
        logger.info("[Rerank] Service terminates. ")