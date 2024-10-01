from passlib.hash import bcrypt

def hash_password(password: str) -> str:
    # Hash the password using bcrypt
    return bcrypt.hash(password)

def verify_hash(plain_password: str, hashed_password: str) -> bool:
    # Verify the plain password against the hashed password using bcrypt
    return bcrypt.verify(plain_password, hashed_password)




