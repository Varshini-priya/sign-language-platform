"""
SignAI Platform — FastAPI Backend
Entry point: uvicorn app.main:app --reload
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings


# ── Lifespan (startup / shutdown) ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # from app.ml.model_loader import load_model
    # app.state.model = load_model(settings.MODEL_PATH)
    
    # Test DB connection on startup
    from app.database import engine
    from sqlalchemy import text
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ Database connected")
    except Exception as e:
        print(f"⚠️  Database not connected: {e}")
    print("✅ SignAI backend started")
    yield
    print("👋 SignAI backend shutting down")

# ── App factory ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SignAI API",
    description="AI-Powered Sign Language Communication Platform",
    version="0.1.0",
    lifespan=lifespan,
)

# ── CORS — allow React dev server ──────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────
from app.routers import auth
app.include_router(auth.router)

# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/health", tags=["system"])
async def health_check():
    return {
        "status": "ok",
        "version": "0.1.0",
        "service": "SignAI Backend",
    }


@app.get("/", tags=["system"])
async def root():
    return {"message": "SignAI API is running. Visit /docs for API reference."}
