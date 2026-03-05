from typing import Annotated, Optional
from fastapi import APIRouter, UploadFile, File, Form
from app.services import face_service

router = APIRouter()

@router.post("/register_data")
def register_data_endpoint(
    nombres: Annotated[str, Form(...)],
    apellidos: Annotated[str, Form(...)],
):
    result = face_service.register_worker_data(nombres=nombres, apellidos=apellidos)
    return result

@router.post("/login_face")
async def login_face_endpoint(
    live_photo: Annotated[UploadFile, File(...)]
):
    photo_bytes = await live_photo.read()
    result = await face_service.process_initial_face_login(photo_bytes)
    return result

@router.post("/update_kyc")
async def update_kyc_endpoint(
    user_id: Annotated[str, Form(...)],
    ci_front: Annotated[UploadFile, File(...)],
    ci_back: Annotated[UploadFile, File(...)],
    license_front: Annotated[Optional[UploadFile], File()] = None,
    license_back: Annotated[Optional[UploadFile], File()] = None
):
    bytes_ci_front = await ci_front.read()
    bytes_ci_back = await ci_back.read()
    bytes_license_front = await license_front.read() if license_front else None
    bytes_license_back = await license_back.read() if license_back else None

    result = await face_service.process_kyc_update(
        user_id=user_id,
        ci_front=bytes_ci_front,
        ci_back=bytes_ci_back,
        license_front=bytes_license_front,
        license_back=bytes_license_back
    )
    
    return result

@router.post("/auth_face")
async def auth_face_endpoint(
    user_id: Annotated[str, Form(...)],
    live_photo: Annotated[UploadFile, File(...)]
):
    photo_bytes = await live_photo.read()
    result = await face_service.process_face_registration(user_id, photo_bytes)
    return result
