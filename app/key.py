
# import os
# from secrets import token_bytes
# from base64 import b64encode


# secret_key = b64encode(token_bytes(64)).decode()
# print(f"Generated Secret Key: {secret_key}")

####################################################################################################################

import bcrypt

# Generate a bcrypt hash for an empty string
bcrypt_hash = bcrypt.hashpw(b"", bcrypt.gensalt())

# Trim the bcrypt hash to 64 characters
secret_key = bcrypt_hash.decode()[:64]

print("Bcrypt Secret Key:", secret_key)
####################################################################################################################