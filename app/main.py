import os
import time
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.models.response import HealthResponse
from app.models.claim import ClaimCase, DecisionResponse, AgentTrace
from app.graph.workflow import ClaimEngine
from app.config import settings
from app.retrieval.hybrid_retriever import build_indices, index_status


app = FastAPI(
    title="Aptino Claim Decision Engine",
    version="1.0.0",
    description="Policy-Aware Multi-Agent RAG Claim Decision Engine",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine: ClaimEngine = None
_readiness = {"ready": False, "detail": "Startup has not completed"}


def get_engine() -> ClaimEngine:
    global _engine
    if _engine is None:
        _engine = ClaimEngine()
    return _engine


@app.on_event("startup")
def startup():
    logging.basicConfig(level=logging.INFO)
    global _readiness
    ready, detail = index_status(settings.POLICY_PDF_PATH, settings.CHROMA_PERSIST_DIR)
    try:
        if not ready:
            logging.info("Building retrieval index: %s", detail)
            build_indices(settings.POLICY_PDF_PATH, settings.CHROMA_PERSIST_DIR)
        _readiness["ready"], _readiness["detail"] = index_status(
            settings.POLICY_PDF_PATH, settings.CHROMA_PERSIST_DIR
        )
    except Exception:
        logging.exception("Retrieval index startup failed")
        _readiness = {"ready": False, "detail": "Policy index initialization failed"}


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy" if _readiness["ready"] else "degraded",
        ready=_readiness["ready"], detail=_readiness["detail"]
    )


@app.get("/")
def root():
    return {
        "service": "Aptino Claim Decision Engine API",
        "health": "/health",
        "analyze": "POST /analyze",
        "web_ui": "http://127.0.0.1:8501",
    }


@app.post("/analyze", response_model=DecisionResponse)
def analyze(case: ClaimCase):
    start = time.time()
    try:
        if not _readiness["ready"]:
            raise HTTPException(status_code=503, detail="Policy index is not ready. Retry shortly.")
        engine = get_engine()
        result = engine.analyze(case)
        result.trace.append(AgentTrace(
            agent="API",
            action="Request completed",
            duration_ms=int((time.time() - start) * 1000),
            evidence_count=len(result.citations),
            metadata={}
        ))
        return result
    except HTTPException:
        raise
    except Exception:
        logging.exception("Claim analysis failed")
        raise HTTPException(status_code=503, detail="Claim analysis is temporarily unavailable.")
