import re
import os
from fastapi import HTTPException , Depends
from app.hashing import hash_password, verify_hash
from app.models import Admin, SecretKey, AdminInfo
from sqlalchemy.exc import IntegrityError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typing import Union, List
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from typing import Union, Optional
from typing import Generator


# Get the database URL from the environment variable
DATABASE_URL = os.environ.get("DATABASE_URL")

# Check if DATABASE_URL is defined
if DATABASE_URL is None:
    raise EnvironmentError("DATABASE_URL environment variable is not defined.")

# Create the SQLAlchemy engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def initialize_database():
    from models import Base
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully.")


# Dependency to get the database session
def get_database() -> Generator[Session, None, None]:
    # Provide a database session to use within the request
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_secret_key(db: Session) -> str:
    try:
        secret_key_record = db.query(SecretKey).first()
        return secret_key_record.key_value if secret_key_record else None
    except Exception as e:
        print(f"Error getting secret key: {e}")
        return None


def is_admin_taken(db: Session, username: str) -> bool:
    try:
        query = db.query(Admin).filter(Admin.username == username).first()
        return query is not None
    except Exception as e:
        print(f"Error checking if admin is taken: {e}")
        return False

def create_admin(db: Session):
    try:
        # Get the secret key from the database
        db_key = get_secret_key(db)
        if db_key is None:
            raise ValueError("Secret key not found in the database.")

        # Validate the input secret key against the database key
        input_key = input("Enter the secret key: ")
        if input_key != db_key:
            raise ValueError("Invalid secret key. Access denied.")

        # Get admin username and password
        username = input("Enter admin username: ")
        password = input("Enter admin password: ")

        # Check if username and password are provided
        if not username or not password:
            print("Both username and password are required.")
            return

        # Check if the username is already taken
        if is_admin_taken(db, username):
            print(f"Username '{username}' is already taken. Please choose a different username.")
            return

        # Validate password format
        if not re.match("^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,16}$", password):
            print("Password must be 8-16 characters long and contain at least one uppercase letter, one lowercase letter, one digit, and one special character.")
            return

        # Hash the password
        hashed_password = hash_password(password.encode())

        # Insert admin details into the database
        admin = Admin(username=username, hashed_password=hashed_password)
        db.add(admin)
        db.commit()

        print(f"Admin account '{username}' created successfully.")

        return {
            "id": admin.id,
            "username": admin.username,
            "hashed_password": admin.hashed_password,
            "admin_flag": admin.admin_flag,
            "created_at": admin.created_at
        }

    except IntegrityError as e:
        print(f"Error creating admin account: {e}")
        db.rollback()

    except ValueError as ve:
        print(ve)

    finally:
        db.close()

if __name__ == "__main__":
    initialize_database()
    db = SessionLocal()
    create_admin(db)


def get_current_admin(db: Session) -> Union[AdminInfo, None]:
    try:
        admin = db.query(Admin).filter(Admin.id == 1).first()
        if not admin:
            return None
        return AdminInfo(
            id=admin.id,
            username=admin.username,
            admin_flag=admin.admin_flag,
            created_at=admin.created_at
        )
    except SQLAlchemyError as e:
        raise RuntimeError(f"Error retrieving admin details: {e}")
    finally:
        db.close()
            

def get_admin(username: str):
    db = SessionLocal()
    try:
        admin = db.query(Admin).filter(Admin.username == username).first()
        if not admin:
            raise HTTPException(status_code=404, detail=f"Admin not found with username '{username}'")
        return admin
    except SQLAlchemyError as e:
        raise RuntimeError(f"Error retrieving admin: {e}")
    finally:
        db.close()



def authenticate_admin(username: str, password: Optional[str] = None, db: Session = Depends(get_database)) -> Union[Admin, None]:
    if password is None:
        return None
    
    try:
        admin = db.query(Admin).filter(Admin.username == username).first()

        if not admin:
            return None

        if not verify_hash(password, admin.hashed_password):
            return None

        return admin
    
    except Exception as e:
        # Handle exceptions here
        pass  # Placeholder for exception handling logic

    return None  # Return None if authentication fails



def get_all_admins() -> List[AdminInfo]:
    db = SessionLocal()
    try:
        admins = db.query(AdminInfo).all()
        if not admins:
            raise ValueError("No admins found")
        return admins
    except SQLAlchemyError as e:
        raise RuntimeError(f"Error retrieving admins: {e}")
    finally:
        db.close()


def verify_admin_password(username: str, current_password: str) -> bool:
    admin = get_admin(username)
    if not admin:
        raise ValueError("Invalid username or password")
    if not verify_hash(current_password, admin.hashed_password):
        raise ValueError("Invalid username or password")
    return True


def update_admin_password(username: str, new_hashed_password: str, db: Session) -> bool:
    try:
        admin = db.query(Admin).filter(Admin.username == username).first()
        if not admin:
            raise ValueError(f"Admin not found with username '{username}'")
        
        admin.hashed_password = new_hashed_password
        db.commit()
        print(f"Admin password updated successfully.")
        return True
    except SQLAlchemyError as e:
        db.rollback()
        raise RuntimeError(f"Error updating admin password: {e}")
    finally:
        db.close()


def delete_admin_by_username(username: str) -> bool:
        admin = get_admin(username)
        if not admin:
            raise HTTPException(status_code=404, detail=f"Admin with username '{username}' not found")

        db = SessionLocal()
        try:
            db.delete(admin)
            db.commit()
            print(f"Admin account '{username}' deleted successfully.")
            return True
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Error deleting admin account: {e}")
        finally:
            db.close()