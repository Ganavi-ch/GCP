from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Coupon(BaseModel):
    code: str
    description: str
    discount_percent: int
    active: bool = True
    expires_at: datetime
    min_order_amount: float = 0
    user_ids: Optional[List[str]] = None


class OrderItem(BaseModel):
    sku: str
    name: str
    quantity: int
    unit_price: float


class Order(BaseModel):
    order_id: str
    user_id: str
    status: str
    created_at: datetime
    estimated_delivery: datetime
    delivery_address: str
    items: List[OrderItem]

    @property
    def total_amount(self) -> float:
        return sum(item.quantity * item.unit_price for item in self.items)


class User(BaseModel):
    user_id: str
    name: str
    email: str
    address: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=2)
    user_id: Optional[str] = None


class ChatResponse(BaseModel):
    intent: str
    confidence: float
    action: str
    data: Dict[str, Any]
    answer: str
