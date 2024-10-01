from datetime import datetime, timedelta, timezone
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status, Form
from .user_database import get_user
from .models import TokenData, User
import os
from typing import Generator, Annotated
from sqlalchemy.orm import Session
from .user_database import SessionLocal

SECRET_KEY = os.environ.get('USER_SECRET_KEY')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 600

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Dependency to get the database session
def get_database() -> Generator[Session, None, None]:
    # Provide a database session to use within the request
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_user_access_token(data: dict, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        # Set default expiration time to 30 minutes if expires_delta is not provided
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: Session = Depends(get_database)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(username=token_data.username, db=db)  # Pass db argument
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(user: User = Depends(get_current_user)) -> User:
    if not user.roles:
        raise HTTPException(status_code=403, detail="User does not have any roles assigned")
    return user
