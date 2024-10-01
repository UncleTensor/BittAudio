# # Import the necessary modules
# import os
# import sys
# from getpass import getpass  # Use getpass to hide the password input
# from fastapi import FastAPI, Request, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from slowapi.errors import RateLimitExceeded
# from fastapi.responses import JSONResponse
# import json

# # Define the function to create the FastAPI application
# def create_app(secret_key: str):
#     # Set the project root path
#     project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

#     # Add the project root directory to the Python path
#     sys.path.insert(0, project_root)

#     # Import routers from the 'routers' package using relative imports
#     from .routers import admin, user, login

#     # Create FastAPI application object
#     app = FastAPI()

#     #Allow CORS for only the React frontend server
#     # origins = [
#     #     "http://85.239.241.96:3000",  # Your React frontend server's HTTP URL
#     #     "http://api.bittaudio.ai",
#     #     "http://144.91.69.154:8000",
#     #     "http://localhost:3000",
#     #     "http:127.0.0.1:3000",
#     #     "http://89.37.121.214:44107",
#     #     "http://149.11.242.18:14428",
#     #     "http://bittaudio.ai",
#     #     "http://v1.bittaudio.ai",
#     #     "http://v2.bittaudio.ai",
#     # ]


#     # # Allow CORS only if not handled by Nginx
#     # app.add_middleware(
#     #     CORSMiddleware,
#     #     allow_origins=origins,
#     #     allow_credentials=True,
#     #     allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
#     #     allow_headers=["*"],
#     #     expose_headers=["*"],
#     # )


#     # Define the list of allowed origins
#     origins = [
#         "http://localhost:3000",
#         "http://127.0.0.1:3000",
#         "https://bittaudio.ai",
#         "https://api.bittaudio.ai",
#         "https://v1.api.bittaudio.ai",
#         "http://149.36.1.168:41981"
#     ]

#     # Register the global exception handler for RateLimitExceeded

#     @app.exception_handler(RateLimitExceeded)
#     async def rate_limit_exception_handler(request: Request, exc: RateLimitExceeded):
#         """
#         Handle RateLimitExceeded exceptions globally and return a JSON response with a custom error message.

#         Args:
#             request (Request): The incoming request causing the rate limit exception.
#             exc (RateLimitExceeded): The RateLimitExceeded exception object.

#         Returns:
#             JSONResponse: JSON response with status code 429 (Too Many Requests) and a custom error message.
#         """
#         print("Oops! You have exceeded the rate limit: 1 request / 5 minutes. Please try again later.")
#         error_message = {"error": "Oops! You have exceeded the rate limit: 1 request / 5 minutes. Please try again later."}
#         return JSONResponse(
#             status_code=429,
#             content=json.loads(json.dumps(error_message))
#     )


#     # Allow CORS for all origins specified in the list
#     app.add_middleware(
#         CORSMiddleware,
#         allow_origins=origins,
#         allow_credentials=True,
#         allow_methods=["GET", "POST", "PUT", "DELETE"],
#         allow_headers=["*"],
#     )

#     # Include routers
#     app.include_router(login.router, prefix="", tags=["Authentication"])
#     app.include_router(admin.router, prefix="", tags=["Admin"])
#     app.include_router(user.router, prefix="", tags=["User"])

#     return app



import os
import sys
from getpass import getpass  # Use getpass to hide the password input
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse
import json

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

# Create a Limiter instance
limiter = Limiter(key_func=get_remote_address)

# Define the function to create the FastAPI application
def create_app(secret_key: str):
    # Set the project root path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Add the project root directory to the Python path
    sys.path.insert(0, project_root)

    # Import routers from the 'routers' package using relative imports
    from .routers import admin, user, login

    # Create FastAPI application object
    app = FastAPI()

    # Allow CORS for only the React frontend server
    origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://bittaudio.ai",
        "https://api.bittaudio.ai",
        "https://v1.api.bittaudio.ai",
        "http://79.117.18.84:38287"
    ]

    # Register the global exception handler for RateLimitExceeded
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_exception_handler(request: Request, exc: RateLimitExceeded):
        """
        Handle RateLimitExceeded exceptions globally and return a JSON response with a custom error message.
        """
        error_message = {"error": "Oops! You have exceeded the rate limit: 1 request / 5 minutes. Please try again later."}
        return JSONResponse(
            status_code=429,
            content=json.loads(json.dumps(error_message))
        )

    # Allow CORS for all origins specified in the list
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(login.router, prefix="", tags=["Authentication"])
    app.include_router(admin.router, prefix="", tags=["Admin"])
    app.include_router(user.router, prefix="", tags=["User"])

    return app
