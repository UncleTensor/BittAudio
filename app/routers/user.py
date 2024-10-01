# user.py
import torchaudio
from fastapi import APIRouter, Depends, HTTPException, Form, Request
from ..user_database import get_user, verify_user_credentials, update_user_password, get_database
from pydantic import BaseModel
import logging
from datetime import datetime
from ..models import User
from ..user_auth import get_current_active_user
import re
from fastapi.encoders import jsonable_encoder
from ..end_points.ttm_api import TTM_API
import bittensor as bt
from fastapi.responses import FileResponse
import os
import random
from sqlalchemy.orm import Session
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi import Limiter


# Create a Limiter instance
limiter = Limiter(key_func=get_remote_address)
router = APIRouter()
ttm_api = TTM_API()



# Define a Pydantic model for the request body
class TTMrequest(BaseModel):
    prompt: str # The prompt for the Text-to-Music service
#     duration: int # The duration of the audio in seconds

@router.post("/change_password", response_model=dict)
async def change_user_password(
    username: str = Form(...),
    current_password: str = Form(...),
    new_password: str = Form(..., min_length=8, max_length=16, regex="^[a-zA-Z0-9!@#$%^&*()_+{}\[\]:;<>,.?/~\\-=|\\\\]+$"),
    confirm_new_password: str = Form(...),
    current_active_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_database)  # Dependency to provide the session
):
    try:
        # Validate that all required fields are provided
        if not username or not current_password or not new_password or not confirm_new_password:
            bt.logging.error("All fields are required.")
            raise HTTPException(status_code=400, detail="All fields are required.")

        # Check if the username exists
        user = get_user(username, db)
        if not user:
            bt.logging.error("User not found.")
            raise HTTPException(status_code=404, detail="User not found")

        # Verify the user's current password
        if not verify_user_credentials(username, current_password, db):
            bt.logging.error("Invalid credentials.")
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Check if the new password and confirm new password match
        if new_password != confirm_new_password:
            bt.logging.error("New password and confirm new password do not match.")
            raise HTTPException(status_code=400, detail="New password and confirm new password do not match.")

        # Check if the new password is different from the current password
        if current_password == new_password:
            bt.logging.error("New password must be different from the current password.")
            raise HTTPException(status_code=400, detail="New password must be different from the current password.")

        # Additional validation: Check if the new password meets the specified conditions
        if not re.match("^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]+$", new_password):
            bt.logging.error("New password must contain at least one uppercase letter, one lowercase letter, one digit, and one special character.")
            raise HTTPException(status_code=400, detail="New password must contain at least one uppercase letter, one lowercase letter, one digit, and one special character.")

        # Update the user's password
        updated_user = update_user_password(username, new_password, db)
        if not updated_user:
            bt.logging.error("Failed to update password.")
            raise HTTPException(status_code=500, detail="Failed to update password.")

        # Return a success message
        bt.logging.success("Password changed successfully")
        return {"message": "Password changed successfully"}

    except HTTPException as e:
        raise e  # Re-raise HTTPException to return specific error response
    except Exception as e:
        # Log the error for debugging
        logging.error(f"Error during password change: {e}")
        # Return a generic error response
        raise HTTPException(status_code=500, detail="Internal Server Error. Check the server logs for more details.")


##########################################################################################################################
# Endpoint for ttm_service
@router.post("/ttm_service")
# @limiter.limit("1/5 minutes")  # Limit to one request per minute per user
async def ttm_service(request: TTMrequest, user: User = Depends(get_current_active_user)):
    try:
        user_dict = jsonable_encoder(user)
        print("User details:", user_dict)

        #check if the user has subscription or not
        if user.roles:
            role = user.roles[0]
            if user.subscription_end_time and datetime.utcnow() <= user.subscription_end_time and role.ttm_enabled == 1:
                print("Congratulations! You have access to Text-to-Music (TTM) service.")

                request_data = request.dict()  # Convert Pydantic model to dictionary
                print('_______________request_data_____________', request_data)

                prompt = request_data.get("prompt")
                duration = request_data.get("duration")

                bt.logging.info("__________request prompt____________: ", prompt)
                bt.logging.info("__________request duration____________: ", duration)

                # Get filtered axons
                filtered_axons = ttm_api.get_filtered_axons()
                bt.logging.info(f"Filtered axons: {filtered_axons}")

                # Check if there are axons available
                if not filtered_axons:
                    bt.logging.error("No axons available for Text-to-Music.")
                    raise HTTPException(status_code=404, detail="No axons available for Text-to-Music.")

                # Choose a TTM axon randomly
                uid, axon = random.choice(filtered_axons)
                bt.logging.info(f"Chosen axon: {axon}, UID: {uid}")
                response = ttm_api.query_network(axon, prompt, duration=duration)

                # Process the response
                audio_data = ttm_api.process_response(axon, response, prompt, api=True)
                bt.logging.info(f"Audio data: {audio_data}")

                try:
                    file_extension = os.path.splitext(audio_data)[1].lower()
                    bt.logging.info(f"audio_file_path: {audio_data}")
                except Exception as e:
                    print(e)
                    bt.logging.error(f"Error processing audio file path or server unaviable for uid: {uid}")
                    raise HTTPException(status_code=404, detail= f"Error processing audio file path or server unavailable for uid: {uid}")
                # Process each audio file path as needed

                if file_extension not in ['.wav', '.mp3']:
                    bt.logging.error(f"Unsupported audio format for uid: {uid}")
                    raise HTTPException(status_code=405, detail="Unsupported audio format.")

                # Set the appropriate content type based on the file extension
                content_type = "audio/wav" if file_extension == '.wav' else "audio/mpeg"

                # Return the audio file
                return FileResponse(path=audio_data, media_type=content_type, filename=os.path.basename(audio_data), headers={"TTM-Axon-UID": str(uid)})

            else:
                print(f"{user.username}! You do not have any access to Text-to-Music (TTM) service or subscription is expired.")
                raise HTTPException(status_code=401, detail=f"{user.username}! Your subscription have been expired or you does not have any access to Text-to-Music (TTM) service")
        else:
            print(f"{user.username}! You do not have any roles assigned.")
            raise HTTPException(status_code=401, detail=f"{user.username}! Your does not have any roles assigned")

    except RateLimitExceeded as e:
        # Handle the RateLimitExceeded exception
        print(f"Rate limit exceeded: {e}")
        raise HTTPException(
            status_code=429,
            detail="Oops! You have exceeded the rate limit: 1 request / 5 minutes. Please try again later.")
