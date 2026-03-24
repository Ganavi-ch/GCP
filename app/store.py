from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from app.models import Coupon, Order, OrderItem, User
from app.sample_data import seed_coupons, seed_orders, seed_users


class DataStore:
    def __init__(self) -> None:
        self.users: dict[str, User] = {u.user_id: u for u in seed_users()}
        self.orders: dict[str, Order] = {o.order_id: o for o in seed_orders()}
        self.coupons: dict[str, Coupon] = {c.code: c for c in seed_coupons()}
        self.sessions: dict[str, dict[str, str]] = {}

    def get_session_ctx(self, user_id: str | None) -> dict[str, str]:
        key = user_id or "guest"
        if key not in self.sessions:
            self.sessions[key] = {}
        return self.sessions[key]

    def register_user(self, name: str, email: str, address: str = "") -> User:
        user_id = f"u{str(uuid4().int)[:6]}"
        user = User(user_id=user_id, name=name, email=email, address=address or "Not provided")
        self.users[user_id] = user
        return user

    def get_user(self, user_id: Optional[str] = None, email: Optional[str] = None) -> Optional[User]:
        if user_id and user_id in self.users:
            return self.users[user_id]
        if email:
            for user in self.users.values():
                if user.email.lower() == email.lower():
                    return user
        return None

    def list_user_orders(self, user_id: str) -> list[Order]:
        return [o for o in self.orders.values() if o.user_id == user_id]

    def get_order(self, order_id: str) -> Optional[Order]:
        return self.orders.get(order_id)

    def cancel_order(self, order_id: str) -> Optional[Order]:
        order = self.orders.get(order_id)
        if not order:
            return None
        if order.status in {"Delivered", "Cancelled"}:
            return order
        order.status = "Cancelled"
        return order

    def update_order_address(self, order_id: str, new_address: str) -> Optional[Order]:
        order = self.orders.get(order_id)
        if not order:
            return None
        order.delivery_address = new_address
        return order

    def update_order_item_quantity(self, order_id: str, sku: str, quantity: int) -> Optional[Order]:
        order = self.orders.get(order_id)
        if not order:
            return None
        for item in order.items:
            if item.sku.lower() == sku.lower():
                item.quantity = quantity
                return order
        return None

    def list_coupons_for_user(self, user_id: Optional[str] = None) -> list[Coupon]:
        now = datetime.now(timezone.utc)
        valid = [c for c in self.coupons.values() if c.active and c.expires_at > now]
        if not user_id:
            return valid
        scoped: list[Coupon] = []
        for coupon in valid:
            if not coupon.user_ids or user_id in coupon.user_ids:
                scoped.append(coupon)
        return scoped

    def get_coupon(self, code: str) -> Optional[Coupon]:
        return self.coupons.get(code.upper())


store = DataStore()
