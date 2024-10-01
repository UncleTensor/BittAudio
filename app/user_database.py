from fastapi import HTTPException, Depends
from sqlalchemy.orm import Session, sessionmaker
import os
from .models import User, Role  # Import correct classes
from sqlalchemy import create_engine
from .hashing import hash_password, verify_hash
import logging
from datetime import datetime, timedelta
from typing import Optional, Union, Generator 
from sqlalchemy.ext.declarative import declarative_base

# Get the database URL from the environment variable
DATABASE_URL = os.environ.get("DATABASE_URL")

# Check if DATABASE_URL is defined
if DATABASE_URL is None:
    raise EnvironmentError("DATABASE_URL environment variable is not defined.")

# Create the SQLAlchemy engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Create a declarative base
Base = declarative_base()


def initialize_database():
    # Create the tables in the database
    User.metadata.create_all(bind=engine)
    Role.metadata.create_all(bind=engine)
    print("Database initialized successfully.")

def get_database() -> Generator[Session, None, None]:
    # Provide a database session to use within the request
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_role_details(role_name):
    # Define the roles with their properties
    roles = [
        {"role_name": "Role_1", "tts_enabled": 1, "ttm_enabled": 1, "vc_enabled": 1},
        {"role_name": "Role_2", "tts_enabled": 1, "ttm_enabled": 0, "vc_enabled": 0},
        {"role_name": "Role_3", "tts_enabled": 0, "ttm_enabled": 1, "vc_enabled": 0},
        {"role_name": "Role_4", "tts_enabled": 0, "ttm_enabled": 0, "vc_enabled": 1},
        {"role_name": "Role_5", "tts_enabled": 1, "ttm_enabled": 1, "vc_enabled": 0},
        {"role_name": "Role_6", "tts_enabled": 0, "ttm_enabled": 1, "vc_enabled": 1},
        {"role_name": "Role_7", "tts_enabled": 1, "ttm_enabled": 0, "vc_enabled": 1},
    ]

    # Find the role details by role_name
    role = next((r for r in roles if r["role_name"] == role_name), None)

    return role

def assign_user_roles(username, selected_role, subscription_duration):
    engine = create_engine(DATABASE_URL)
    connection = engine.connect()
    transaction = connection.begin()

    try:
        # Fetch the role details based on the selected_role
        role_details = get_role_details(selected_role)

        if role_details is None:
            raise HTTPException(status_code=400, detail=f"Invalid role: {selected_role}")

        # Calculate subscription start and end time based on subscription duration
        subscription_start_time = datetime.utcnow()
        subscription_end_time = subscription_start_time + timedelta(days=subscription_duration)

        # Fetch the existing roles of the user
        existing_roles = connection.execute('SELECT role_name, tts_enabled, ttm_enabled, vc_enabled, subscription_end_time FROM roles WHERE username = %s', (username,))
        existing_roles = existing_roles.fetchall()

        # Check if the user already has roles assigned
        if existing_roles:
            # Create a dictionary to map existing roles with their details
            existing_roles_dict = {row[0]: {'tts_enabled': row[1], 'ttm_enabled': row[2], 'vc_enabled': row[3], 'subscription_end_time': row[4]} for row in existing_roles}

            # Update the existing_roles_dict with the new role details
            existing_roles_dict[selected_role] = {'tts_enabled': role_details["tts_enabled"], 'ttm_enabled': role_details["ttm_enabled"], 'vc_enabled': role_details["vc_enabled"], 'subscription_end_time': subscription_end_time}

            # Clear existing roles for the user
            connection.execute('DELETE FROM roles WHERE username = %s', (username,))

            # Insert the updated roles into the 'roles' table
            for role_name, details in existing_roles_dict.items():
                connection.execute('''
                    INSERT INTO roles (username, role_name, tts_enabled, ttm_enabled, vc_enabled, subscription_start_time, subscription_end_time)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                ''', (username, role_name, details['tts_enabled'], details['ttm_enabled'], details['vc_enabled'], subscription_start_time, details['subscription_end_time']))

            print(f"Roles updated successfully for user {username}!")
        else:
            # Insert the new role into the 'roles' table
            connection.execute('''
                INSERT INTO roles (username, role_name, tts_enabled, ttm_enabled, vc_enabled, subscription_start_time, subscription_end_time)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            ''', (username, selected_role, role_details["tts_enabled"], role_details["ttm_enabled"], role_details["vc_enabled"], subscription_start_time, subscription_end_time))

            print(f"Role '{selected_role}' assigned to user {username} successfully!")

        # Commit the transaction
        transaction.commit()

    except HTTPException as he:
        # Handle invalid role error
        print(f"Error assigning role: {he}")
        transaction.rollback()
        raise he
    except Exception as e:
        # Handle any errors that may occur during database operations
        print(f"Error assigning roles: {e}")
        transaction.rollback()
        raise HTTPException(status_code=500, detail="Internal Server Error")
    finally:
        # Close the connection
        connection.close()

    
    
def create_user(username, password, role_assign, subscription_duration_in_days):
    try:
        with SessionLocal() as session:
            # Check if both username and password are provided
            if not username or not password:
                raise HTTPException(status_code=400, detail="Both username and password are required.")

            # Check if the username is already taken
            user = session.query(User).filter(User.username == username).first()
            if user:
                raise HTTPException(status_code=400, detail=f"Username '{username}' is already taken. Please choose a different username.")

            # Validate the role syntax before proceeding
            role_details = get_role_details(role_assign)
            if role_details is None:
                raise HTTPException(status_code=400, detail=f"Invalid role syntax: {role_assign}")

            # Calculate subscription end time based on subscription duration
            subscription_end_time = datetime.utcnow() + timedelta(days=subscription_duration_in_days)

            # Create User instance
            new_user = User(username=username, hashed_password=hash_password(password), subscription_end_time=subscription_end_time)
            session.add(new_user)

            # Create Role instance
            new_role = Role(username=username, role_name=role_assign, tts_enabled=role_details["tts_enabled"],
                            ttm_enabled=role_details["ttm_enabled"], vc_enabled=role_details["vc_enabled"],
                            subscription_end_time=subscription_end_time)
            session.add(new_role)

            # Commit the changes
            session.commit()

            print("User created successfully with roles.")

            # Return user details
            return {
                "id": new_user.id,
                "username": new_user.username,
                "created_at": new_user.created_at,
                "subscription_end_time": new_user.subscription_end_time,
                "roles": [{"role_name": new_role.role_name, "tts_enabled": new_role.tts_enabled,
                        "ttm_enabled": new_role.ttm_enabled, "vc_enabled": new_role.vc_enabled}]
            }

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error creating user account: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")






def get_user(username: str, db: Session) -> Union[User, None]:
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return None
    return user

def authenticate_user(username: str, password: Optional[str] = None, db: Session = Depends(get_database)) -> Union[User, None]:
    if password is None:
        return None
    
    user = get_user(username, db=db)

    if not user:
        return None

    if not verify_hash(password, user.hashed_password):
        return None

    return user



def verify_user_credentials(username: str, plain_text_password: str, db: Session) -> bool:
    user = db.query(User).filter(User.username == username).first()
    if not user:
        logging.error(f"User '{username}' not found.")
        return False
    return verify_hash(plain_text_password, user.hashed_password)



def update_user_password(username: str, new_plain_text_password: str, db: Session) -> bool:
    user = db.query(User).filter(User.username == username).first()
    if not user:
        logging.error(f"User '{username}' not found.")
        return False

    new_hashed_password = hash_password(new_plain_text_password)
    user.hashed_password = new_hashed_password
    db.commit()
    return True

