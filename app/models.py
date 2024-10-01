from sqlalchemy import Column, Integer, String, DateTime, Boolean, Table, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from pydantic import BaseModel
from sqlalchemy.sql import func

Base = declarative_base()

class Admin(Base):
    __tablename__ = "admins"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    admin_flag = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow())

class SecretKey(Base):
    __tablename__ = "secret_keys"

    id = Column(Integer, primary_key=True, index=True)
    key_value = Column(String, unique=True, index=True)



class Role(Base):
    __tablename__ = 'roles'

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, ForeignKey('users.username'))  # Foreign key referencing User table
    role_name = Column(String)
    tts_enabled = Column(Integer)
    ttm_enabled = Column(Integer)
    vc_enabled = Column(Integer)
    subscription_start_time = Column(DateTime, default=datetime.utcnow())
    subscription_end_time = Column(DateTime)

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow())
    subscription_end_time = Column(DateTime)
    
    roles = relationship("Role", back_populates="user")

# Add back reference to Role
Role.user = relationship("User", back_populates="roles")

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str | None = None

class AdminCreate(BaseModel):
    username: str
    password: str

class AdminUpdate(BaseModel):
    current_password: str
    new_password: str

class AdminInfo(BaseModel):
    id: int
    username: str

class RoleAssignment(BaseModel):
    username: str
    role_name: str

class AdminBase(BaseModel):
    id: int
    username: str
    admin_flag: int
    created_at: datetime

class SecretKeyBase(BaseModel):
    id: int
    key_value: str
