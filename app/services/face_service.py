import cv2
import numpy as np
import asyncio
import onnxruntime as ort
from fastapi import HTTPException
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop
from app.db.firebase import db
from google.cloud import firestore

face_app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'landmark_2d_106'], providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

ruta_liveness = "models/modelrgb.onnx"
try:
    liveness_session = ort.InferenceSession(ruta_liveness, providers=['CPUExecutionProvider'])
    #liveness_session = ort.InferenceSession(ruta_liveness, providers=['CUDAExecutionProvider'])
    liveness_input_name = liveness_session.get_inputs()[0].name
except Exception as e:
    print(f"⚠️ Error cargando modelo Liveness: {e}")

# C. Modelo Reconocimiento Facial (ArcFace ResNet100 MIT)
ruta_arcface = "models/arcfaceresnet100-8.onnx"
try:
    arcface_session = ort.InferenceSession(ruta_arcface, providers=['CPUExecutionProvider'])
    #arcface_session = ort.InferenceSession(ruta_arcface, providers=['CUDAExecutionProvider'])
    arcface_input_name = arcface_session.get_inputs()[0].name
except Exception as e:
    print(f"⚠️ Error cargando modelo ArcFace: {e}")

SIMILARITY_THRESHOLD = 0.48
LIVENESS_THRESHOLD = 0.85

def _analizar_rostro_completo(photo_bytes: bytes):
    nparr = np.frombuffer(photo_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img_bgr is None: raise ValueError("No se pudo decodificar la imagen.")

    faces = face_app.get(img_bgr)
    if len(faces) == 0: raise ValueError("No se detectó ningún rostro.")
    if len(faces) > 1: raise ValueError("Se detectó más de un rostro. Solo debe haber una persona.")
        
    face = faces[0]
    bbox = face.bbox
    
    alto_img, ancho_img = img_bgr.shape[:2]
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    
    face_w, face_h = x2 - x1, y2 - y1
    cx, cy = x1 + face_w // 2, y1 + face_h // 2
    
    scale = 1.5
    side = int(max(face_w, face_h) * scale)
    
    nx1, ny1 = max(0, cx - side // 2), max(0, cy - side // 2)
    nx2, ny2 = min(ancho_img, cx + side // 2), min(alto_img, cy + side // 2)
    
    rostro_recortado = img_bgr[ny1:ny2, nx1:nx2]
    if rostro_recortado.size == 0: raise ValueError("Error al recortar el rostro detectado.")

    img_rgb_liveness = cv2.cvtColor(rostro_recortado, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb_liveness, (112, 112))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_normalized = np.transpose(img_normalized, (2, 0, 1))
    img_normalized = np.expand_dims(img_normalized, axis=0)

    resultado_liveness = liveness_session.run(None, {liveness_input_name: img_normalized})
    score_real = float(resultado_liveness[0][0][1])
    
    is_real = score_real >= LIVENESS_THRESHOLD

    if not is_real:
        return {
            "liveness_score": score_real,
            "is_real": False,
            "embedding": None
        }

    if face.kps is None:
        raise ValueError("No se encontraron puntos de referencia (landmarks) en el rostro.")
    
    rostro_alineado = norm_crop(img_bgr, landmark=face.kps, image_size=112)
    
    arcface_rgb = cv2.cvtColor(rostro_alineado, cv2.COLOR_BGR2RGB)
    arcface_tensor = np.transpose(arcface_rgb, (2, 0, 1))
    arcface_tensor = np.expand_dims(arcface_tensor, axis=0)
    arcface_tensor = (arcface_tensor.astype(np.float32) - 127.5) / 127.5
    
    resultado_arcface = arcface_session.run(None, {arcface_input_name: arcface_tensor})
    vector_crudo = resultado_arcface[0][0]
    
    vector_normalizado = vector_crudo / np.linalg.norm(vector_crudo)
    embedding = [float(x) for x in vector_normalizado]
    
    return {
        "liveness_score": score_real,
        "is_real": True,
        "embedding": embedding
    }

def _calcular_similitud(embedding1: list, embedding2: list) -> float:
    vec1, vec2 = np.array(embedding1), np.array(embedding2)
    dot_product = np.dot(vec1, vec2)
    return float(dot_product / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

async def process_registration(user_id: str, photo_bytes: bytes) -> dict:
    try:
        analisis = await asyncio.to_thread(_analizar_rostro_completo, photo_bytes)
        
        if not analisis["is_real"]:
            db.collection('fraud_attempts').add({
                'user_id': user_id, 'action': 'register', 
                'liveness_score': analisis["liveness_score"], 'timestamp': firestore.SERVER_TIMESTAMP
            })
            raise HTTPException(status_code=403, detail="Alerta de Seguridad: Se detectó un posible ataque de suplantación (Spoofing).")

        driver_ref = db.collection('drivers').document(user_id)
        driver_ref.set({
            'status': 'registered',
            'face_embedding': analisis["embedding"],
            'created_at': firestore.SERVER_TIMESTAMP
        })
        
        return {
            "status": "success",
            "message": f"Conductor {user_id} registrado exitosamente.",
            "embedding_size": len(analisis["embedding"]),
            "liveness_score": round(analisis["liveness_score"], 4)
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


async def process_verification(user_id: str, photo_bytes: bytes) -> dict:
    try:
        driver_ref = db.collection('drivers').document(user_id)
        doc = driver_ref.get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Usuario no encontrado. Debe registrarse.")
            
        saved_embedding = doc.to_dict().get('face_embedding')
        if not saved_embedding:
            raise HTTPException(status_code=400, detail="El usuario no tiene un rostro registrado.")
            
        analisis = await asyncio.to_thread(_analizar_rostro_completo, photo_bytes)
        
        if not analisis["is_real"]:
            db.collection('fraud_attempts').add({
                'user_id': user_id, 'action': 'verify', 
                'liveness_score': analisis["liveness_score"], 'timestamp': firestore.SERVER_TIMESTAMP
            })
            raise HTTPException(status_code=403, detail="Acceso Denegado: Prueba de vida fallida.")
        
        similarity = _calcular_similitud(saved_embedding, analisis["embedding"])
        is_verified = similarity >= SIMILARITY_THRESHOLD
        
        if is_verified:
            driver_ref.update({
                'last_verified': firestore.SERVER_TIMESTAMP,
                'status': 'active'
            })
            mensaje = "Acceso permitido por 24 horas."
        else:
            mensaje = "Denegado. El rostro no coincide con el conductor registrado."
            
        return {
            "status": "success" if is_verified else "failed",
            "message": mensaje,
            "verified": is_verified,
            "similarity_score": round(similarity, 4),
            "liveness_score": round(analisis["liveness_score"], 4)
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")