from typing import Annotated
from fastapi import APIRouter, UploadFile, File, Form
from app.schemas.user import VerifyResponse
from app.services import face_service

router = APIRouter()

@router.post("/auth", response_model=VerifyResponse)
async def smart_auth_endpoint(
    document_number: Annotated[str, Form(...)], # Aquí Flutter enviará el CI o Pasaporte
    live_photo: Annotated[UploadFile, File(...)]
):
    photo_bytes = await live_photo.read()
    result = await face_service.process_smart_auth(document_number, photo_bytes)
    return result

@router.post("/update_kyc")
async def update_kyc_endpoint(
    user_id: Annotated[str, Form(...)],
    nombres: Annotated[str, Form(...)],
    apellidos: Annotated[str, Form(...)],
    ci_front: Annotated[UploadFile, File(...)],
    ci_back: Annotated[UploadFile, File(...)],
    passport_front: Annotated[UploadFile, File(...)],
    passport_back: Annotated[UploadFile, File(...)]
):
    # Leemos los bytes de todos los archivos que llegaron desde Flutter
    bytes_ci_front = await ci_front.read()
    bytes_ci_back = await ci_back.read()
    bytes_passport_front = await passport_front.read()
    bytes_passport_back = await passport_back.read()

    # Mandamos todo al servicio
    result = await face_service.process_kyc_update(
        user_id=user_id,
        nombres=nombres,
        apellidos=apellidos,
        ci_front=bytes_ci_front,
        ci_back=bytes_ci_back,
        passport_front=bytes_passport_front,
        passport_back=bytes_passport_back
    )
    
    return result