from datetime import datetime, timedelta, timezone

from app.models import Coupon, Order, OrderItem, User


def seed_users() -> list[User]:
    return [
        User(
            user_id="u1001",
            name="John",
            email="john@example.com",
            address="12 Park Street, Bangalore",
        ),
        User(
            user_id="u1002",
            name="Asha",
            email="asha@example.com",
            address="44 MG Road, Chennai",
        ),
    ]


def seed_coupons() -> list[Coupon]:
    now = datetime.now(timezone.utc)
    return [
        Coupon(
            code="SAVE20",
            description="20% off on orders above 1000",
            discount_percent=20,
            active=True,
            expires_at=now + timedelta(days=15),
            min_order_amount=1000,
        ),
        Coupon(
            code="FESTIVE10",
            description="10% festive discount",
            discount_percent=10,
            active=False,
            expires_at=now - timedelta(days=2),
            min_order_amount=500,
        ),
        Coupon(
            code="VIP30",
            description="30% for premium users",
            discount_percent=30,
            active=True,
            expires_at=now + timedelta(days=30),
            user_ids=["u1001"],
        ),
    ]


def seed_orders() -> list[Order]:
    now = datetime.now(timezone.utc)
    return [
        Order(
            order_id="12345",
            user_id="u1001",
            status="Shipped",
            created_at=now - timedelta(days=3),
            estimated_delivery=now + timedelta(days=2),
            delivery_address="12 Park Street, Bangalore",
            items=[
                OrderItem(sku="SKU100", name="Running Shoes", quantity=1, unit_price=3200),
                OrderItem(sku="SKU201", name="Socks Pack", quantity=2, unit_price=250),
            ],
        ),
        Order(
            order_id="10234",
            user_id="u1002",
            status="Processing",
            created_at=now - timedelta(days=1),
            estimated_delivery=now + timedelta(days=4),
            delivery_address="44 MG Road, Chennai",
            items=[
                OrderItem(sku="SKU302", name="Bluetooth Speaker", quantity=1, unit_price=2500),
            ],
        ),
    ]
