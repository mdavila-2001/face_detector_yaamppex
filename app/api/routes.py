from fastapi import APIRouter, UploadFile, File, Form
from app.schemas.user import VerifyResponse
from app.services import face_service

router = APIRouter()

@router.post("/auth", response_model=VerifyResponse)
async def smart_auth_endpoint(
    document_number: str = Form(...), # Aquí Flutter enviará el CI o Pasaporte
    live_photo: UploadFile = File(...)
):
    photo_bytes = await live_photo.read()
    result = await face_service.process_smart_auth(document_number, photo_bytes)
    return result