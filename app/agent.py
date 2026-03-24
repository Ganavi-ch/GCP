from __future__ import annotations

import re
from datetime import datetime, timezone

from app.knowledge_base import rag_answer
from app.store import store


def _get_ctx(user_id: str | None) -> dict[str, str]:
    return store.get_session_ctx(user_id)


def _extract_order_id(text: str) -> str | None:
    match = re.search(r"order\s*(?:id)?\s*(\d{4,})", text.lower())
    return match.group(1) if match else None


def _extract_coupon_code(text: str) -> str | None:
    match = re.search(r"coupon\s+([a-z0-9]+)", text.lower())
    return match.group(1).upper() if match else None


def _extract_email(text: str) -> str | None:
    match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    return match.group(0) if match else None


def _extract_name(text: str) -> str | None:
    match = re.search(
        r"name\s+([a-zA-Z ]+?)\s+(?:and\s+)?(?:email|e-mail)\s+[\w\.-]+@[\w\.-]+\.\w+",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).strip().title()

    # Fallback for messages like "Register me with name John"
    match = re.search(r"name\s+([a-zA-Z ]{2,40})", text, flags=re.IGNORECASE)
    return match.group(1).strip().title() if match else None


def _extract_address(text: str) -> str | None:
    match = re.search(r"address(?:\s+to)?\s+(.+)$", text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else None


def _extract_sku(text: str) -> str | None:
    match = re.search(r"(sku[0-9]+)", text.lower())
    return match.group(1).upper() if match else None


def _extract_quantity(text: str) -> int | None:
    match = re.search(r"quantity(?:\s+to)?\s+(\d+)", text.lower())
    return int(match.group(1)) if match else None


def _extract_address_after_order(text: str, order_id: str) -> str | None:
    lower = text.lower()
    idx = lower.find(order_id.lower())
    if idx == -1:
        return None
    tail = text[idx + len(order_id) :].strip()
    if not tail:
        return None
    # Trim common separators left behind by natural language.
    tail = re.sub(r"^[\s\-\:,.]+", "", tail)
    return tail if len(tail) >= 6 else None


def detect_intent(message: str, user_id: str | None = None) -> tuple[str, float]:
    text = message.lower()
    has_email = bool(re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text))
    if "available coupon" in text or "coupons are available" in text or "show" in text and "coupon" in text and "available" in text:
        return "coupon_list", 0.95
    if "coupon" in text and ("status" in text or "details" in text or "check" in text or "info" in text):
        return "coupon_detail", 0.95
    # Follow-up for coupon
    if ("details" in text or "status" in text or "check" in text or "info" in text) and "it" in text and _get_ctx(user_id).get("last_coupon_code"):
        return "coupon_detail", 0.85

    if "cancel" in text and "order" in text:
        return "order_cancel", 0.97
    if "cancel" in text and ("it" in text or "my order" in text or "this order" in text):
        return "order_cancel", 0.86
    if ("change" in text or "update" in text) and ("address" in text or "delivery" in text):
        return "order_update_address", 0.92
    if ("update" in text or "change" in text or "set" in text) and ("quantity" in text or "qty" in text):
        return "order_update_quantity", 0.92

    # Follow-ups for order
    if ("status" in text or "when" in text or "arrive" in text or "deliver" in text or "track" in text) and ("it" in text or "my order" in text or "status" in text) and _get_ctx(user_id).get("last_order_id"):
        return "order_status", 0.88
    if ("details" in text or "show" in text or "info" in text) and ("it" in text or "this order" in text) and _get_ctx(user_id).get("last_order_id"):
        return "order_detail", 0.88

    if ("order" in text and "status" in text) or "when will" in text and "delivered" in text:
        return "order_status", 0.93
    if ("deliver" in text or "delivery" in text or "track" in text) and ("order" in text or "my order" in text or "it" in text or "this" in text):
        return "order_status", 0.82
    if ("order" in text and "details" in text) or ("show" in text and "order" in text) or ("tell" in text and "order" in text):
        return "order_detail", 0.93
    if ("show" in text or "details" in text or "info" in text) and ("order" in text or "my order" in text or "it" in text or "this" in text) and "coupon" not in text:
        return "order_detail", 0.78
    # Treat "register/create account" as an action unless it's clearly a "how-to" question.
    # Use word boundary so "registered" doesn't trigger the action.
    is_how_question = text.startswith("how ") or text.startswith("how can")
    has_register_action = (
        bool(re.search(r"\bregister\b", text))
        or "register me" in text
        or "new account" in text
        or "create a new account" in text
        or (has_email and ("name" in text or "my name" in text))
    )
    if has_register_action and (has_email or not is_how_question):
        return "user_register", 0.92

    # If it's just an email and we were waiting for it
    if has_email and _get_ctx(user_id).get("state") == "awaiting_registration":
        return "user_register", 0.95
    if "profile" in text or "registered address" in text or "user details" in text:
        return "user_detail", 0.9
    if "order history" in text:
        return "user_order_history", 0.91
    if "what" in text and ("can you do" in text or "are you" in text or "help" in text):
        return "agent_help", 0.95
    return "knowledge_qa", 0.72


def run_agent(message: str, user_id: str | None = None) -> dict:
    # Plan A: try Vertex AI agent first.
    # We only fall back if Vertex is not configured or specifically disabled.
    try:
        from app.vertex_agent import run_vertex_agent
        return run_vertex_agent(message, user_id)
    except (ImportError, RuntimeError, ValueError) as e:
        # Fallback to local deterministic agent only if Vertex isn't available/configured.
        # Log the error for visibility
        print(f"Vertex AI Agent unavailable, falling back to regex: {e}")
        pass
    except Exception as e:
        # For other unexpected errors, we might still want to fall back to keep the service alive,
        # but the goal is to make Vertex reliable.
        print(f"Unexpected error in Vertex Agent: {e}")
        pass

    intent, confidence = detect_intent(message, user_id)
    text = message.strip()
    ctx = _get_ctx(user_id)

    # Contextual fallbacks to support follow-ups like "cancel it" or "when will it arrive?"
    if intent in {"order_status", "order_detail", "order_cancel", "order_update_address", "order_update_quantity"}:
        if ("order" not in text.lower()) or ("order id" not in text.lower()):
            # If user didn't specify an id, try context.
            if not _extract_order_id(text) and ctx.get("last_order_id"):
                # We'll override later by using ctx when order_id is missing.
                pass

    if intent in {"coupon_detail"}:
        if not _extract_coupon_code(text) and ctx.get("last_coupon_code"):
            pass

    if intent == "coupon_list":
        coupons = store.list_coupons_for_user(user_id)
        data = {"coupons": [c.model_dump() for c in coupons]}
        if coupons:
            lines = []
            for c in coupons:
                expires = c.expires_at.date().isoformat()
                lines.append(f"- {c.code}: {c.discount_percent}% off (min order {c.min_order_amount}, expires {expires})")
            answer = "Available coupons for you:\n" + "\n".join(lines)
        else:
            answer = "No active coupons found for your account."
        return {"intent": intent, "confidence": confidence, "action": "list_coupons", "data": data, "answer": answer}

    if intent == "coupon_detail":
        code = _extract_coupon_code(text)
        if not code and ctx.get("last_coupon_code"):
            code = ctx["last_coupon_code"]
        if not code:
            return {"intent": intent, "confidence": 0.4, "action": "none", "data": {}, "answer": "Please share the coupon code."}
        coupon = store.get_coupon(code)
        if not coupon:
            return {"intent": intent, "confidence": confidence, "action": "get_coupon", "data": {}, "answer": f"Coupon {code} not found."}
        status = "active" if coupon.active and coupon.expires_at > datetime.now(timezone.utc) else "inactive/expired"
        restricted = bool(coupon.user_ids) and (user_id is not None) and (user_id not in coupon.user_ids)
        scope = "Restricted to specific users." if restricted else "Available for you."
        data = {"coupon": coupon.model_dump(), "status": status}
        answer = (
            f"Coupon {code} is {status}.\n"
            f"Description: {coupon.description}\n"
            f"Discount: {coupon.discount_percent}%\n"
            f"Minimum order amount: {coupon.min_order_amount}\n"
            f"Expires: {coupon.expires_at.date().isoformat()}\n"
            f"{scope}"
        )
        ctx["last_coupon_code"] = code
        return {"intent": intent, "confidence": confidence, "action": "get_coupon", "data": data, "answer": answer}

    if intent in {"order_status", "order_detail", "order_cancel", "order_update_address", "order_update_quantity"}:
        order_id = _extract_order_id(text)
        if not order_id and ctx.get("last_order_id"):
            order_id = ctx["last_order_id"]
        if not order_id:
            if intent == "order_status" and user_id:
                orders = store.list_user_orders(user_id)
                if not orders:
                    return {"intent": intent, "confidence": 0.5, "action": "none", "data": {}, "answer": "No orders found for your account. Please provide an order id."}
                latest = sorted(orders, key=lambda x: x.created_at, reverse=True)[0]
                order_id = latest.order_id
            else:
                return {"intent": intent, "confidence": 0.45, "action": "none", "data": {}, "answer": "Please provide a valid order id."}
        order = store.get_order(order_id)
        if not order:
            return {"intent": intent, "confidence": confidence, "action": "get_order", "data": {}, "answer": f"Order {order_id} not found."}
        ctx["last_order_id"] = order_id

        if intent == "order_status":
            answer = (
                f"Order {order_id}\n"
                f"Status: {order.status}\n"
                f"Estimated delivery: {order.estimated_delivery.date().isoformat()}\n"
                f"Delivery address: {order.delivery_address}"
            )
            return {"intent": intent, "confidence": confidence, "action": "get_order_status", "data": {"order": order.model_dump()}, "answer": answer}
        if intent == "order_detail":
            items_lines = []
            for item in order.items:
                line_total = item.quantity * item.unit_price
                items_lines.append(f"- {item.sku} ({item.name}): {item.quantity} x {item.unit_price} = {line_total}")
            answer = (
                f"Order details for {order_id}\n"
                f"Status: {order.status}\n"
                f"Created: {order.created_at.date().isoformat()}\n"
                f"Estimated delivery: {order.estimated_delivery.date().isoformat()}\n"
                f"Delivery address: {order.delivery_address}\n"
                f"Items:\n" + "\n".join(items_lines) + "\n"
                f"Total amount: {order.total_amount}"
            )
            return {"intent": intent, "confidence": confidence, "action": "get_order_details", "data": {"order": order.model_dump()}, "answer": answer}
        if intent == "order_cancel":
            existing = store.get_order(order_id)
            updated = store.cancel_order(order_id)
            if not updated:
                return {
                    "intent": intent,
                    "confidence": confidence,
                    "action": "cancel_order",
                    "data": {},
                    "answer": f"Order {order_id} not found.",
                }
            if existing and existing.status in {"Delivered", "Cancelled"}:
                return {
                    "intent": intent,
                    "confidence": confidence,
                    "action": "cancel_order",
                    "data": {"order": updated.model_dump()},
                    "answer": f"Order {order_id} cannot be cancelled because it is already {existing.status}.",
                }
            return {
                "intent": intent,
                "confidence": confidence,
                "action": "cancel_order",
                "data": {"order": updated.model_dump() if updated else {}},
                "answer": f"Order {order_id} has been cancelled. Current status: {updated.status}.",
            }
        if intent == "order_update_address":
            new_address = _extract_address_after_order(text, order_id) or _extract_address(text)
            if not new_address:
                return {"intent": intent, "confidence": 0.55, "action": "none", "data": {}, "answer": "Please provide the new delivery address."}
            updated = store.update_order_address(order_id, new_address)
            return {
                "intent": intent,
                "confidence": confidence,
                "action": "update_order_address",
                "data": {"order": updated.model_dump() if updated else {}},
                "answer": f"Updated delivery address for order {order_id}.\nNew address: {updated.delivery_address if updated else new_address}",
            }
        if intent == "order_update_quantity":
            existing_order = store.get_order(order_id)
            if not existing_order or not existing_order.items:
                return {"intent": intent, "confidence": 0.5, "action": "none", "data": {}, "answer": f"Order {order_id} not found or has no items."}

            sku = _extract_sku(text)
            if not sku:
                sku = existing_order.items[0].sku

            qty = _extract_quantity(text)
            if qty is None:
                current_item = next((i for i in existing_order.items if i.sku.lower() == sku.lower()), None)
                current_qty = current_item.quantity if current_item else "unknown"
                return {
                    "intent": intent,
                    "confidence": 0.55,
                    "action": "none",
                    "data": {},
                    "answer": f"Which quantity would you like to set for {sku} in order {order_id}? Current quantity is {current_qty}.",
                }

            updated = store.update_order_item_quantity(order_id, sku, qty)
            if not updated:
                return {"intent": intent, "confidence": 0.5, "action": "none", "data": {}, "answer": f"SKU {sku} not found in order {order_id}."}

            updated_item = next((i for i in updated.items if i.sku.lower() == sku.lower()), None)
            new_qty = updated_item.quantity if updated_item else qty
            ctx["last_sku"] = sku
            return {
                "intent": intent,
                "confidence": confidence,
                "action": "update_order_item_quantity",
                "data": {"order": updated.model_dump(), "sku": sku, "quantity": new_qty},
                "answer": f"Updated {sku} quantity to {new_qty} for order {order_id}.",
            }

    if intent == "user_register":
        email = _extract_email(text)
        name = _extract_name(text) or "New User"
        if not email:
            ctx["state"] = "awaiting_registration"
            return {
                "intent": intent,
                "confidence": 0.45,
                "action": "none",
                "data": {},
                "answer": "Please provide a valid email to register.",
            }
        user = store.register_user(name=name, email=email)
        ctx["state"] = None  # Clear state
        return {
            "intent": intent,
            "confidence": confidence,
            "action": "register_user",
            "data": {"user": user.model_dump()},
            "answer": f"User registered successfully with id {user.user_id}.",
        }

    if intent == "user_detail":
        email = _extract_email(text)
        user = store.get_user(user_id=user_id, email=email)
        if not user:
            return {"intent": intent, "confidence": 0.5, "action": "get_user", "data": {}, "answer": "User not found. Share user id or email."}
        lower = text.lower()
        if "registered address" in lower or (("address" in lower) and ("registered" in lower or "my address" in lower)):
            return {
                "intent": intent,
                "confidence": confidence,
                "action": "get_user",
                "data": {"user": user.model_dump()},
                "answer": f"Your registered address is: {user.address}",
            }
        return {
            "intent": intent,
            "confidence": confidence,
            "action": "get_user",
            "data": {"user": user.model_dump()},
            "answer": f"Profile details:\nName: {user.name}\nEmail: {user.email}\nAddress: {user.address}",
        }

    if intent == "user_order_history":
        if not user_id:
            return {"intent": intent, "confidence": 0.5, "action": "none", "data": {}, "answer": "Please provide user_id for order history."}
        orders = store.list_user_orders(user_id)
        if not orders:
            return {
                "intent": intent,
                "confidence": confidence,
                "action": "list_user_orders",
                "data": {"orders": []},
                "answer": f"No orders found for user {user_id}.",
            }

        lines = []
        for o in sorted(orders, key=lambda x: x.created_at, reverse=True):
            lines.append(
                f"- {o.order_id}: {o.status}, delivered on/est. {o.estimated_delivery.date().isoformat()}, total {o.total_amount}"
            )
        return {
            "intent": intent,
            "confidence": confidence,
            "action": "list_user_orders",
            "data": {"orders": [o.model_dump() for o in orders]},
            "answer": f"Order history for user {user_id} (found {len(orders)}):\n" + "\n".join(lines),
        }

    if intent == "agent_help":
        return {
            "intent": intent,
            "confidence": confidence,
            "action": "none",
            "data": {},
            "answer": "Hi! I am your E-Commerce AI Assistant. Ask me about coupons, orders, profile, or platform help.",
        }

    kb = rag_answer(text)
    if "register" in kb["source"].lower():
        ctx["state"] = "awaiting_registration"

    return {
        "intent": "knowledge_qa",
        "confidence": confidence,
        "action": "rag_answer",
        "data": {"source": kb["source"]},
        "answer": kb["answer"],
    }
