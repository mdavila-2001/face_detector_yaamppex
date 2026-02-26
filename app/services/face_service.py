import cv2
import numpy as np
import asyncio
from fastapi import HTTPException
from insightface.app import FaceAnalysis
from app.db.firebase import db
from google.cloud import firestore
import onnxruntime as ort

# ==========================================
# 1. INICIALIZACIÓN DEL MODELO (Una sola vez)
# ==========================================
# Cargamos el modelo en memoria al arrancar el servidor, no en cada petición.
# 'buffalo_l' es un modelo excelente y balanceado de InsightFace.
# Usamos CPUExecutionProvider para tus pruebas locales. Cuando pases al server con GPU,
# solo cambiarás esto a ['CUDAExecutionProvider']
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Umbral de similitud (Threshold). Arriba de 0.60 (60%) suele ser la misma persona en InsightFace.
SIMILARITY_THRESHOLD = 0.51 

# ==========================================
# 2. FUNCIONES SINCRONAS (Trabajo pesado de CPU)
# ==========================================
def _procesar_imagen_y_extraer_vector(photo_bytes: bytes):
    """
    Decodifica la imagen con OpenCV y usa InsightFace para extraer el embedding.
    Esta función es pesada, por eso la ejecutaremos en un hilo separado.
    """
    # Convertir los bytes a un arreglo de numpy para OpenCV
    nparr = np.frombuffer(photo_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("No se pudo decodificar la imagen. Formato inválido.")

    # Detectar rostros
    faces = face_app.get(img)
    
    if len(faces) == 0:
        raise ValueError("No se detectó ningún rostro en la imagen.")
    if len(faces) > 1:
        raise ValueError("Se detectó más de un rostro. Por seguridad, solo debe haber una persona.")
        
    # Extraer el vector (embedding) de 512 números del primer rostro detectado
    embedding = faces[0].embedding
    
    # Convertirlo a lista de floats estándar de Python para poder guardarlo en Firebase
    return [float(x) for x in embedding]

def _calcular_similitud(embedding1: list, embedding2: list) -> float:
    """
    Calcula la Similitud Coseno entre dos vectores. 
    Devuelve un valor entre -1 y 1. Mientras más cerca de 1, más se parecen.
    """
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return float(similarity)

# ==========================================
# 3. ENDPOINTS ASÍNCRONOS (Rutas FastAPI)
# ==========================================
async def process_registration(user_id: str, photo_bytes: bytes) -> dict:
    try:
        # Aquí iría tu validación de Liveness en el futuro
        
        # Ejecutamos el modelo de IA en un HILO SEPARADO para no bloquear FastAPI
        embedding = await asyncio.to_thread(_procesar_imagen_y_extraer_vector, photo_bytes)
        
        # Guardar en Firebase
        driver_ref = db.collection('drivers').document(user_id)
        driver_ref.set({
            'status': 'registered',
            'face_embedding': embedding,
            'created_at': firestore.SERVER_TIMESTAMP
        })
        
        return {
            "status": "success",
            "message": f"Rostro del usuario {user_id} analizado y registrado correctamente.",
            "embedding_size": len(embedding)
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

async def process_verification(user_id: str, photo_bytes: bytes) -> dict:
    try:
        # 1. Recuperar vector maestro de Firebase
        driver_ref = db.collection('drivers').document(user_id)
        doc = driver_ref.get()
        
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Usuario no encontrado. Debe registrarse.")
            
        driver_data = doc.to_dict()
        saved_embedding = driver_data.get('face_embedding')
        
        if not saved_embedding:
            raise HTTPException(status_code=400, detail="El usuario no tiene un rostro registrado.")
            
        # 2. Extraer vector de la foto actual (EN HILO SEPARADO)
        current_embedding = await asyncio.to_thread(_procesar_imagen_y_extraer_vector, photo_bytes)
        
        # 3. Comparar matemáticamente
        similarity = _calcular_similitud(saved_embedding, current_embedding)
        
        # 4. Decidir si es la misma persona
        is_verified = similarity >= SIMILARITY_THRESHOLD
        
        if is_verified:
            driver_ref.update({
                'last_verified': firestore.SERVER_TIMESTAMP,
                'status': 'active',
                'last_score': float(similarity)
            })
            mensaje = "Acceso permitido."
        else:
            db.collection('failed_attempts').add({
                'user_id': user_id,
                'score': float(similarity),
                'timestamp': firestore.SERVER_TIMESTAMP
            })
            mensaje = "Denegado. El rostro no coincide con el registrado."
            
        return {
            "status": "success" if is_verified else "failed",
            "message": mensaje,
            "verified": is_verified,
            "similarity_score": round(similarity, 4)
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")