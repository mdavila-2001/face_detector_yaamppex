import cv2
import numpy as np
import asyncio
import onnxruntime as ort
from fastapi import HTTPException
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop
from app.db.firebase import db
from google.cloud import firestore

# ==========================================
# 1. INICIALIZACIÓN DE MODELOS EN MEMORIA
# ==========================================
# A. Modelo de Identidad (InsightFace - Solo Detección y Landmarks)
face_app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'landmark_2d_106'], providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# B. Modelo Anti-Spoofing / Liveness (MiniFASNet)
ruta_liveness = "models/modelrgb.onnx"
try:
    liveness_session = ort.InferenceSession(ruta_liveness, providers=['CPUExecutionProvider'])
    liveness_input_name = liveness_session.get_inputs()[0].name
except Exception as e:
    print(f"⚠️ Error cargando modelo Liveness: {e}")

# C. Modelo Reconocimiento Facial (ArcFace ResNet100 MIT)
ruta_arcface = "models/arcfaceresnet100-8.onnx"
try:
    arcface_session = ort.InferenceSession(ruta_arcface, providers=['CPUExecutionProvider'])
    arcface_input_name = arcface_session.get_inputs()[0].name
except Exception as e:
    print(f"⚠️ Error cargando modelo ArcFace: {e}")

# Umbrales de Seguridad
SIMILARITY_THRESHOLD = 0.48  # Para conductores con variaciones de luz/look
LIVENESS_THRESHOLD = 0.85    # 85% de certeza mínima de que está vivo

# ==========================================
# 2. FUNCIONES SINCRONAS (Trabajo pesado CPU)
# ==========================================
def _analizar_rostro_completo(photo_bytes: bytes):
    """
    Decodifica la imagen, extrae el rostro, verifica Liveness y devuelve el Vector.
    Todo en un solo flujo optimizado.
    """
    nparr = np.frombuffer(photo_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        raise ValueError("No se pudo decodificar la imagen.")

    # 1. Detectar el rostro y extraer features con InsightFace
    faces = face_app.get(img_bgr)
    
    if len(faces) == 0:
        raise ValueError("No se detectó ningún rostro.")
    if len(faces) > 1:
        raise ValueError("Se detectó más de un rostro. Solo debe haber una persona.")
        
    face = faces[0]
    bbox = face.bbox # Coordenadas: [x1, y1, x2, y2]
    
    # === 2A. Recortar e inferir Reconocimiento Facial (ArcFace MIT) ===
    # InsightFace nos da los 5 puntos clave (ojos, nariz, comisuras) en face.kps
    if face.kps is None:
        raise ValueError("No se encontraron puntos de referencia (landmarks) en el rostro.")
    
    # norm_crop alinea matemáticamente la cara basándose en los landmarks
    rostro_alineado = norm_crop(img_bgr, landmark=face.kps, image_size=112)
    
    # Preparar el tensor para ArcFace (espera RGB, float32, y normalizado ((pixels - 127.5) / 127.5))
    arcface_rgb = cv2.cvtColor(rostro_alineado, cv2.COLOR_BGR2RGB)
    arcface_tensor = np.transpose(arcface_rgb, (2, 0, 1)) # (3, 112, 112)
    arcface_tensor = np.expand_dims(arcface_tensor, axis=0) # (1, 3, 112, 112)
    arcface_tensor = (arcface_tensor.astype(np.float32) - 127.5) / 127.5
    
    # Ejecutar ArcFace
    resultado_arcface = arcface_session.run(None, {arcface_input_name: arcface_tensor})
    embedding = [float(x) for x in resultado_arcface[0][0]] # Vector de 512 dimensiones
    
    # === 2B. Recortar el rostro para el test de Liveness (expandido) ===
    # El modelo MiniFASNet necesita contexto (fondo/bordes) para saber si es una foto real o un teléfono.
    alto_img, ancho_img = img_bgr.shape[:2]
    x1, y1 = int(bbox[0]), int(bbox[1])
    x2, y2 = int(bbox[2]), int(bbox[3])
    
    face_w = x2 - x1
    face_h = y2 - y1
    cx = x1 + face_w // 2
    cy = y1 + face_h // 2
    
    # Expandimos el recorte un 50% (escala 1.5)
    scale = 1.5
    side = int(max(face_w, face_h) * scale)
    
    nx1 = max(0, cx - side // 2)
    ny1 = max(0, cy - side // 2)
    nx2 = min(ancho_img, cx + side // 2)
    ny2 = min(alto_img, cy + side // 2)
    
    rostro_recortado = img_bgr[ny1:ny2, nx1:nx2]
    
    if rostro_recortado.size == 0:
        raise ValueError("Error al recortar el rostro detectado.")

    # 3. Preparar imagen para el modelo Anti-Spoofing (Requiere RGB y 112x112)
    img_rgb = cv2.cvtColor(rostro_recortado, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (112, 112))
    
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_normalized = np.transpose(img_normalized, (2, 0, 1))
    img_normalized = np.expand_dims(img_normalized, axis=0)

    # 4. Ejecutar Anti-Spoofing
    resultado_liveness = liveness_session.run(None, {liveness_input_name: img_normalized})
    score_real = float(resultado_liveness[0][0][1]) # Probabilidad de ser "Real"
    
    # 5. Devolvemos el resultado empacado
    return {
        "liveness_score": score_real,
        "is_real": score_real >= LIVENESS_THRESHOLD,
        "embedding": embedding
    }

def _calcular_similitud(embedding1: list, embedding2: list) -> float:
    vec1, vec2 = np.array(embedding1), np.array(embedding2)
    dot_product = np.dot(vec1, vec2)
    return float(dot_product / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

# ==========================================
# 3. ENDPOINTS ASÍNCRONOS (Rutas FastAPI)
# ==========================================
async def process_registration(user_id: str, photo_bytes: bytes) -> dict:
    try:
        # Enviamos el trabajo pesado al hilo secundario
        analisis = await asyncio.to_thread(_analizar_rostro_completo, photo_bytes)
        
        # FILTRO DE SEGURIDAD 🛡️
        if not analisis["is_real"]:
            db.collection('fraud_attempts').add({
                'user_id': user_id, 'action': 'register', 
                'liveness_score': analisis["liveness_score"], 'timestamp': firestore.SERVER_TIMESTAMP
            })
            raise HTTPException(status_code=403, detail="Alerta de Seguridad: Se detectó un posible ataque de suplantación (Spoofing).")

        # Guardar en Firebase
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
        # 1. Recuperar vector de Firebase
        driver_ref = db.collection('drivers').document(user_id)
        doc = driver_ref.get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Usuario no encontrado. Debe registrarse.")
            
        saved_embedding = doc.to_dict().get('face_embedding')
        if not saved_embedding:
            raise HTTPException(status_code=400, detail="El usuario no tiene un rostro registrado.")
            
        # 2. Analizar foto actual
        analisis = await asyncio.to_thread(_analizar_rostro_completo, photo_bytes)
        
        # FILTRO DE SEGURIDAD 🛡️
        if not analisis["is_real"]:
            db.collection('fraud_attempts').add({
                'user_id': user_id, 'action': 'verify', 
                'liveness_score': analisis["liveness_score"], 'timestamp': firestore.SERVER_TIMESTAMP
            })
            raise HTTPException(status_code=403, detail="Acceso Denegado: Prueba de vida fallida.")
        
        # 3. Comparar matemáticamente
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