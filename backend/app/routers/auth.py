"""
auth_router.py — Week 4
FastAPI router: register, login, refresh, me.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
#from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User
from app.auth_schemas import (
    UserRegister, UserLogin, TokenResponse, UserResponse,
    RefreshRequest, AccessTokenResponse,
)
from app.auth_service import register_user, authenticate_user, generate_token_pair
from app.security import create_access_token, get_user_id_from_token

router = APIRouter(prefix="/api/auth", tags=["auth"])
#oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")
bearer_scheme = HTTPBearer()


'''def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    user_id = get_user_id_from_token(token, expected_type="access")
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User not found")
    return user'''

def get_current_user(
        credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme), 
        db: Session = Depends(get_db)
    ) -> User:
    
    token = credentials.credentials 
    
    user_id = get_user_id_from_token(token, expected_type="access")
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User not found")
    
    return user


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(payload: UserRegister, db: Session = Depends(get_db)):
    try:
        user = register_user(db, payload)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    return user


@router.post("/login", response_model=TokenResponse)
def login(payload: UserLogin, db: Session = Depends(get_db)):
    user = authenticate_user(db, payload.email, payload.password)
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password")
    access_token, refresh_token = generate_token_pair(user)
    return TokenResponse(access_token=access_token, refresh_token=refresh_token, user=user)


@router.post("/refresh", response_model=AccessTokenResponse)
def refresh(payload: RefreshRequest):
    user_id = get_user_id_from_token(payload.refresh_token, expected_type="refresh")
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired refresh token")
    new_access_token = create_access_token(user_id)
    return AccessTokenResponse(access_token=new_access_token)


@router.get("/me", response_model=UserResponse)
def me(current_user: User = Depends(get_current_user)):
    return current_user
