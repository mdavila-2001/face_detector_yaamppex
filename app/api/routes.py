from fastapi import APIRouter, UploadFile, File, Form
from app.schemas.user import RegisterResponse, VerifyResponse
from app.services import face_service

router = APIRouter()

@router.post("/register", response_model=RegisterResponse)
async def register_endpoint(
    user_id: str = Form(...),
    live_photo: UploadFile = File(...)
):
    photo_bytes = await live_photo.read()
    result = await face_service.process_registration(user_id, photo_bytes)
    return result

@router.post("/verify", response_model=VerifyResponse)
async def verify_endpoint(
    user_id: str = Form(...),
    live_photo: UploadFile = File(...)
):
    photo_bytes = await live_photo.read()
    result = await face_service.process_verification(user_id, photo_bytes)
    return result