from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import IntegrityError
from datetime import datetime
import os


DATABASE_URL = os.environ.get("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def initialize_database():
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully.")

class SecretKey(Base):
    __tablename__ = "secret_keys"

    id = Column(Integer, primary_key=True, index=True)
    key_value = Column(String, unique=True, index=True)


def add_secret_key(secret_key):
    try:
        session = SessionLocal()
        db_key = SecretKey(key_value=secret_key)
        session.add(db_key)
        session.commit()
        print("Secret key added successfully.")
    except IntegrityError:
        session.rollback()
        print("Error: Secret key already exists.")
    except Exception as e:
        session.rollback()
        print(f"Error adding secret key: {e}")
    finally:
        session.close()


def update_secret_key(secret_key):
    try:
        session = SessionLocal()
        db_key = session.query(SecretKey).first()
        db_key.key_value = secret_key
        session.commit()
        print("Secret key updated successfully.")
    except Exception as e:
        session.rollback()
        print(f"Error updating secret key: {e}")
    finally:
        session.close()


if __name__ == "__main__":
    initialize_database()
    action = input("Enter 'add' to add a new secret key or 'update' to update an existing secret key: ")

    if action == 'add':
        secret_key = input("Enter the secret key to add: ")
        add_secret_key(secret_key)
    elif action == 'update':
        secret_key = input("Enter the new secret key: ")
        update_secret_key(secret_key)
    else:
        print("Invalid action. Please enter 'add' or 'update'.")