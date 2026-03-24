from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agent import run_agent
from app.models import ChatRequest, ChatResponse
from app.store import store

app = FastAPI(
    title="Agentic E-Commerce Assistant",
    version="1.0.0",
    description="GCP-ready Agentic AI backend for e-commerce operations and knowledge Q&A.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Agentic E-Commerce Assistant API is running."}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    result = run_agent(request.message, request.user_id)
    return ChatResponse(**result)


@app.get("/users/{user_id}")
def get_user(user_id: str):
    user = store.get_user(user_id=user_id)
    return {"user": user.model_dump() if user else None}


@app.get("/orders/{order_id}")
def get_order(order_id: str):
    order = store.get_order(order_id)
    return {"order": order.model_dump() if order else None}


@app.get("/coupons")
def list_coupons(user_id: str | None = None):
    coupons = store.list_coupons_for_user(user_id)
    return {"coupons": [c.model_dump() for c in coupons]}
