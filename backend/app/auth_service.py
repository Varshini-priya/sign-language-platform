"""
auth_service.py — Week 4
Business logic layer for authentication — keeps routers thin.
"""
from sqlalchemy.orm import Session
from app.models import User
from app.security import hash_password, verify_password, create_access_token, create_refresh_token
from app.auth_schemas import UserRegister


def register_user(db: Session, payload: UserRegister) -> User:
    existing = db.query(User).filter(
        (User.email == payload.email) | (User.username == payload.username)
    ).first()
    if existing:
        raise ValueError("Email or username already registered")

    user = User(
        email=payload.email,
        username=payload.username,
        password_hash=hash_password(payload.password[:72]),
        full_name=payload.full_name,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, email: str, password: str) -> User | None:
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.password_hash):
        return None
    return user


def generate_token_pair(user: User) -> tuple[str, str]:
    access_token = create_access_token(str(user.id))
    refresh_token = create_refresh_token(str(user.id))
    return access_token, refresh_token
