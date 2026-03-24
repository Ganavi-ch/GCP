from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Optional

from app.knowledge_base import rag_answer
from app.store import store

# Vertex AI imports are optional at runtime (local fallback).
try:
    import vertexai
    from vertexai.generative_models import (
        Content,
        FunctionDeclaration,
        GenerationConfig,
        GenerativeModel,
        Part,
        Tool,
    )
except Exception:  # pragma: no cover
    vertexai = None


def _get_ctx(user_id: str | None) -> dict[str, str]:
    return store.get_session_ctx(user_id)


def _safe_date(value: Any) -> str:
    try:
        return value.date().isoformat()
    except Exception:
        return str(value)


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
    match = re.search(r"name\s+([a-zA-Z ]{2,40})", text, flags=re.IGNORECASE)
    return match.group(1).strip().title() if match else None


def _extract_address(text: str) -> str | None:
    match = re.search(r"address(?:\s+to)?\s+(.+)$", text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else None


def _action_list_coupons(*, user_id: Optional[str] = None) -> dict[str, Any]:
    coupons = store.list_coupons_for_user(user_id)
    data = {"coupons": [c.model_dump() for c in coupons]}
    if coupons:
        lines = []
        for c in coupons:
            lines.append(
                f"- {c.code}: {c.discount_percent}% off (min order {c.min_order_amount}, expires {c.expires_at.date().isoformat()})"
            )
        answer = "Available coupons for you:\n" + "\n".join(lines)
    else:
        answer = "No active coupons found for your account."
    return {"answer": answer, "data": data}


def _action_get_coupon(*, code: Optional[str] = None, user_id: Optional[str] = None) -> dict[str, Any]:
    ctx = _get_ctx(user_id)
    if not code and ctx.get("last_coupon_code"):
        code = ctx["last_coupon_code"]
    if not code:
        return {"answer": "Please share the coupon code.", "data": {}}
    code = code.upper()
    coupon = store.get_coupon(code)
    if not coupon:
        return {"answer": f"Coupon {code} not found.", "data": {}}

    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    status = "active" if coupon.active and coupon.expires_at > now else "inactive/expired"
    restricted = bool(coupon.user_ids) and (user_id is not None) and (user_id not in (coupon.user_ids or []))
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
    return {"answer": answer, "data": data}


def _action_get_order_status(*, order_id: Optional[str] = None, user_id: Optional[str] = None) -> dict[str, Any]:
    ctx = _get_ctx(user_id)
    if not order_id and ctx.get("last_order_id"):
        order_id = ctx["last_order_id"]
    if not order_id and user_id:
        orders = store.list_user_orders(user_id)
        if orders:
            order_id = sorted(orders, key=lambda o: o.created_at, reverse=True)[0].order_id
    if not order_id:
        return {"answer": "Please provide a valid order id.", "data": {}}

    order = store.get_order(order_id)
    if not order:
        return {"answer": f"Order {order_id} not found.", "data": {}}
    ctx["last_order_id"] = order_id
    answer = (
        f"Order {order_id}\n"
        f"Status: {order.status}\n"
        f"Estimated delivery: {_safe_date(order.estimated_delivery)}\n"
        f"Delivery address: {order.delivery_address}"
    )
    return {"answer": answer, "data": {"order": order.model_dump()}}


def _action_get_order_details(*, order_id: Optional[str] = None, user_id: Optional[str] = None) -> dict[str, Any]:
    ctx = _get_ctx(user_id)
    if not order_id and ctx.get("last_order_id"):
        order_id = ctx["last_order_id"]
    if not order_id and user_id:
        orders = store.list_user_orders(user_id)
        if orders:
            order_id = sorted(orders, key=lambda o: o.created_at, reverse=True)[0].order_id
    if not order_id:
        return {"answer": "Please provide a valid order id.", "data": {}}

    order = store.get_order(order_id)
    if not order:
        return {"answer": f"Order {order_id} not found.", "data": {}}
    ctx["last_order_id"] = order_id

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
    return {"answer": answer, "data": {"order": order.model_dump()}}


def _action_cancel_order(*, order_id: Optional[str] = None, user_id: Optional[str] = None) -> dict[str, Any]:
    ctx = _get_ctx(user_id)
    if not order_id and ctx.get("last_order_id"):
        order_id = ctx["last_order_id"]
    if not order_id:
        return {"answer": "Please provide a valid order id.", "data": {}}
    existing = store.get_order(order_id)
    if not existing:
        return {"answer": f"Order {order_id} not found.", "data": {}}
    updated = store.cancel_order(order_id)
    if existing.status in {"Delivered", "Cancelled"}:
        return {"answer": f"Order {order_id} cannot be cancelled because it is already {existing.status}.", "data": {}}
    if not updated:
        return {"answer": f"Order {order_id} not found.", "data": {}}
    ctx["last_order_id"] = order_id
    return {"answer": f"Order {order_id} has been cancelled. Current status: {updated.status}.", "data": {"order": updated.model_dump()}}


def _action_update_order_address(*, order_id: Optional[str] = None, new_address: Optional[str] = None, user_id: Optional[str] = None) -> dict[str, Any]:
    ctx = _get_ctx(user_id)
    if not order_id and ctx.get("last_order_id"):
        order_id = ctx["last_order_id"]
    if not order_id:
        return {"answer": "Please provide a valid order id.", "data": {}}
    if not new_address:
        return {"answer": "Please provide the new delivery address.", "data": {}}
    updated = store.update_order_address(order_id, new_address)
    if not updated:
        return {"answer": f"Order {order_id} not found.", "data": {}}
    ctx["last_order_id"] = order_id
    return {"answer": f"Updated delivery address for order {order_id}.\nNew address: {updated.delivery_address}", "data": {"order": updated.model_dump()}}


def _action_update_order_quantity(
    *,
    order_id: Optional[str] = None,
    sku: Optional[str] = None,
    quantity: Optional[int] = None,
    user_id: Optional[str] = None,
) -> dict[str, Any]:
    ctx = _get_ctx(user_id)
    if not order_id and ctx.get("last_order_id"):
        order_id = ctx["last_order_id"]
    if not order_id:
        return {"answer": "Please provide a valid order id.", "data": {}}

    order = store.get_order(order_id)
    if not order or not order.items:
        return {"answer": f"Order {order_id} not found or has no items.", "data": {}}

    if not sku:
        sku = order.items[0].sku
    if quantity is None:
        current_item = next((i for i in order.items if i.sku.lower() == sku.lower()), None)
        current_qty = current_item.quantity if current_item else "unknown"
        return {"answer": f"Which quantity would you like to set for {sku} in order {order_id}? Current quantity is {current_qty}.", "data": {}}

    updated = store.update_order_item_quantity(order_id, sku, quantity)
    if not updated:
        return {"answer": f"SKU {sku} not found in order {order_id}.", "data": {}}
    ctx["last_order_id"] = order_id
    ctx["last_sku"] = sku
    updated_item = next((i for i in updated.items if i.sku.lower() == sku.lower()), None)
    new_qty = updated_item.quantity if updated_item else quantity
    return {"answer": f"Updated {sku} quantity to {new_qty} for order {order_id}.", "data": {"order": updated.model_dump(), "sku": sku, "quantity": new_qty}}


def _action_register_user(*, name: Optional[str] = None, email: Optional[str] = None, address: Optional[str] = None) -> dict[str, Any]:
    if not email:
        return {"answer": "Please provide a valid email to register.", "data": {}}
    if not name:
        name = "New User"
    user = store.register_user(name=name, email=email, address=address or "")
    return {"answer": f"User registered successfully with id {user.user_id}.", "data": {"user": user.model_dump()}}


def _action_get_user_profile(*, user_id: Optional[str] = None, email: Optional[str] = None, show_address_only: bool = False) -> dict[str, Any]:
    user = store.get_user(user_id=user_id, email=email)
    if not user:
        return {"answer": "User not found. Share user id or email.", "data": {}}
    if show_address_only:
        return {"answer": f"Your registered address is: {user.address}", "data": {"user": user.model_dump()}}
    return {
        "answer": f"Profile details:\nName: {user.name}\nEmail: {user.email}\nAddress: {user.address}",
        "data": {"user": user.model_dump()},
    }


def _action_list_order_history(*, user_id: Optional[str] = None) -> dict[str, Any]:
    if not user_id:
        return {"answer": "Please provide user_id for order history.", "data": {}}
    orders = store.list_user_orders(user_id)
    if not orders:
        return {"answer": f"No orders found for user {user_id}.", "data": {"orders": []}}
    lines = []
    for o in sorted(orders, key=lambda x: x.created_at, reverse=True):
        lines.append(f"- {o.order_id}: {o.status}, delivered on/est. {o.estimated_delivery.date().isoformat()}, total {o.total_amount}")
    return {
        "answer": f"Order history for user {user_id} (found {len(orders)}):\n" + "\n".join(lines),
        "data": {"orders": [o.model_dump() for o in orders]},
    }


def _action_rag_answer(*, query: str) -> dict[str, Any]:
    # RAG knowledge base (local fallback); Vertex embeddings upgrade lives in knowledge_base.
    kb = rag_answer(query)
    return {"answer": kb["answer"], "data": {"source": kb["source"]}}


def _tool_map() -> dict[str, Callable[..., dict[str, Any]]]:
    return {
        "list_coupons": _action_list_coupons,
        "get_coupon": _action_get_coupon,
        "get_order_status": _action_get_order_status,
        "get_order_details": _action_get_order_details,
        "cancel_order": _action_cancel_order,
        "update_order_address": _action_update_order_address,
        "update_order_quantity": _action_update_order_quantity,
        "register_user": _action_register_user,
        "get_user_profile": _action_get_user_profile,
        "list_order_history": _action_list_order_history,
        "rag_answer": _action_rag_answer,
    }


def _tools_declarations() -> list[Tool]:
    def decl(name: str, description: str, schema: dict[str, Any]) -> FunctionDeclaration:
        return FunctionDeclaration(name=name, description=description, parameters=schema)

    return [
        Tool(
            function_declarations=[
                decl(
                    "list_coupons",
                    "Retrieve a list of available discount coupons. Use this when the user asks for coupons or promo codes.",
                    {"type": "object", "properties": {"user_id": {"type": "string", "description": "Unique identifier for the user."}}},
                ),
                decl(
                    "get_coupon",
                    "Fetch detailed information about a specific coupon, including its status and discount percentage.",
                    {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "The coupon code (e.g., 'SAVE20')."},
                            "user_id": {"type": "string", "description": "The user ID to check eligibility."},
                        },
                    },
                ),
                decl(
                    "get_order_status",
                    "Check the current processing or shipping status and estimated delivery for an order.",
                    {
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string", "description": "The unique order ID."},
                            "user_id": {"type": "string", "description": "The user ID associated with the order."},
                        },
                    },
                ),
                decl(
                    "get_order_details",
                    "Get a complete breakdown of an order, including item lists, pricing, and delivery address.",
                    {
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string", "description": "The order ID to query."},
                            "user_id": {"type": "string", "description": "The user ID associated with the order."},
                        },
                    },
                ),
                decl(
                    "cancel_order",
                    "Request cancellation of an existing order. This is only possible if the order hasn't been shipped or delivered.",
                    {
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string", "description": "The ID of the order to cancel."},
                            "user_id": {"type": "string", "description": "The user ID associated with the order."},
                        },
                    },
                ),
                decl(
                    "update_order_address",
                    "Change the delivery destination address for an active order.",
                    {
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string", "description": "The order ID to update."},
                            "new_address": {"type": "string", "description": "The complete new delivery address."},
                            "user_id": {"type": "string", "description": "The user ID associated with the order."},
                        },
                    },
                ),
                decl(
                    "update_order_quantity",
                    "Modify the quantity of a specific item (SKU) within an order.",
                    {
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string", "description": "The order ID to update."},
                            "sku": {"type": "string", "description": "The stock keeping unit identifier for the item."},
                            "quantity": {"type": "integer", "description": "The new quantity (must be positive)."},
                            "user_id": {"type": "string", "description": "The user ID associated with the order."},
                        },
                    },
                ),
                decl(
                    "register_user",
                    "Create a new user profile with name, email, and optional address.",
                    {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "The user's full name."},
                            "email": {"type": "string", "description": "The user's email address (must be valid)."},
                            "address": {"type": "string", "description": "The physical delivery address."},
                        },
                        "required": ["name", "email"],
                    },
                ),
                decl(
                    "get_user_profile",
                    "Retrieve the user's registration data, including their registered name, email, and address.",
                    {
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string", "description": "The unique user identifier."},
                            "email": {"type": "string", "description": "The registered email address."},
                            "show_address_only": {"type": "boolean", "description": "If true, only return the address."},
                        },
                    },
                ),
                decl(
                    "list_order_history",
                    "Retrieve a list of all historical orders associated with a user account.",
                    {"type": "object", "properties": {"user_id": {"type": "string", "description": "The unique user identifier."}}},
                ),
                decl(
                    "rag_answer",
                    "Look up platform policies, how-to guides, and general informational queries using the internal knowledge base. Use this for 'how' questions.",
                    {"type": "object", "properties": {"query": {"type": "string", "description": "The user's question or search query."}}},
                ),
            ]
        )
    ]


def run_vertex_agent(message: str, user_id: str | None = None) -> dict[str, Any]:
    if vertexai is None:
        raise RuntimeError("vertexai SDK not available.")

    project_id = os.getenv("VERTEX_PROJECT_ID", "project-ca436020-0c75-4cc6-b84")
    location = os.getenv("VERTEX_LOCATION", "asia-mumbai")
    # Vertex AI uses region identifiers (e.g. asia-south1 for Mumbai).
    if location in ["asia-mumbai", "mumbai"]:
        location = "asia-south1"
    model_name = os.getenv("VERTEX_MODEL_NAME", "gemini-2.5-flash")

    vertexai.init(project=project_id, location=location)
    model = GenerativeModel(model_name=model_name)

    ctx = _get_ctx(user_id)
    memory_summary = {
        "last_order_id": ctx.get("last_order_id"),
        "last_coupon_code": ctx.get("last_coupon_code"),
        "state": ctx.get("state"),
    }

    sys_prompt = (
        "You are an expert E-Commerce AI Assistant. Your goal is to help users with orders, coupons, and account details. "
        "You have access to real-time tools to check and modify the backend state. "
        "### Guidelines:\n"
        "1. **Use Tools Proactively**: If a user asks about an order, coupon, or their profile, ALWAYS call the corresponding tool first. Do not guess.\n"
        "2. **Natural Follow-ups**: If a user refers to 'it', 'that', or 'my order', check the 'Server memory' below. If you see a 'last_order_id', use it for the tool call unless they specify a different one.\n"
        "3. **Handle Missing Info**: If a tool call needs a parameter (like an order_id or email) that is neither in the query nor in the 'Server memory', ask the user for it politely instead of failing.\n"
        "4. **RAG for How-To**: For informational questions like 'How do I cancel?' or 'How can I apply a coupon?', use the 'rag_answer' tool.\n"
        "5. **Flexibility**: Understand varied phrasing. 'Where is my stuff?' should trigger 'get_order_status'.\n"
        "6. **Consistency**: If the user asks 'What can you do?', respond with: 'Hi! I am your E-Commerce AI Assistant. Ask me about coupons, orders, profile, or platform help.'\n\n"
        f"### Server memory for user_id '{user_id}':\n{json.dumps(memory_summary, indent=2)}\n\n"
        "Respond in a helpful, concise manner. Never output raw JSON."
    )

    tools = _tools_declarations()
    tool_map = _tool_map()

    generation_config = GenerationConfig(temperature=0)

    contents: list[Content] = [
        Content(role="user", parts=[Part.from_text(f"{sys_prompt}\n\nUser message: {message}")])
    ]

    max_turns = 3
    final_text: str | None = None
    last_tool_data: dict[str, Any] = {}

    for _ in range(max_turns):
        resp = model.generate_content(contents=contents, tools=tools, generation_config=generation_config)
        if not resp.candidates:
            break
        candidate = resp.candidates[0]
        # If the model requested function calls, execute them.
        fn_calls = getattr(candidate, "function_calls", None) or []
        if not fn_calls:
            final_text = resp.text
            break

        function_response_parts = []
        for fn_call in fn_calls:
            fn_name = fn_call.name
            fn_args = dict(fn_call.args) if getattr(fn_call, "args", None) else {}
            handler = tool_map.get(fn_name)
            if not handler:
                tool_result = {"answer": f"Tool {fn_name} not implemented.", "data": {}}
            else:
                # Inject user_id when schema allows it by the model's omission.
                if "user_id" in handler.__code__.co_varnames and "user_id" not in fn_args:  # type: ignore[attr-defined]
                    fn_args["user_id"] = user_id
                tool_result = handler(**fn_args)  # type: ignore[misc]
                last_tool_data = tool_result

            function_response_parts.append(
                Part.from_function_response(name=fn_name, response={"contents": tool_result})
            )

        function_response_contents = Content(role="user", parts=function_response_parts)
        # Append the model's function-call message + tool responses back to the conversation.
        contents.append(candidate.content)
        contents.append(function_response_contents)

    if not final_text:
        # Fallback: if model failed to produce final text, use last tool response.
        final_text = last_tool_data.get("answer") if last_tool_data else "I couldn't complete that request."

    return {
        "intent": "llm_function_call",
        "confidence": 1.0,
        "action": "tool_agent",
        "data": last_tool_data.get("data", {}) if last_tool_data else {},
        "answer": final_text,
    }

