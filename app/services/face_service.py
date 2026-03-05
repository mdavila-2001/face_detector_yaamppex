import cv2
import numpy as np
import asyncio
import time
import onnxruntime as ort
from fastapi import HTTPException
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop
from app.db.firebase import db
from google.cloud import firestore
import uuid
from firebase_admin import storage

_embedding_cache = {}
_CACHE_TTL = 300

face_app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'landmark_2d_106'], providers=['CPUExecutionProvider'])
#face_app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'landmark_2d_106'], providers=['CUDAExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

ruta_liveness = "models/modelrgb.onnx"
try:
    liveness_session = ort.InferenceSession(ruta_liveness, providers=['CPUExecutionProvider'])
    #liveness_session = ort.InferenceSession(ruta_liveness, providers=['CUDAExecutionProvider'])
    liveness_input_name = liveness_session.get_inputs()[0].name
except Exception as e:
    print(f"⚠️ Error cargando modelo Liveness: {e}")

ruta_arcface = "models/arcfaceresnet100-8.onnx"
try:
    arcface_session = ort.InferenceSession(ruta_arcface, providers=['CPUExecutionProvider'])
    #arcface_session = ort.InferenceSession(ruta_arcface, providers=['CUDAExecutionProvider'])
    arcface_input_name = arcface_session.get_inputs()[0].name
except Exception as e:
    print(f"⚠️ Error cargando modelo ArcFace: {e}")

LIVENESS_THRESHOLD = 0.85
SIMILARITY_THRESHOLD = 0.48

def _analizar_rostro_completo(photo_bytes: bytes, check_liveness: bool = True):
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

    score_real = 0.0
    is_real = True

    if check_liveness:
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

import urllib.request
def register_worker_data(nombres: str, apellidos: str) -> dict:
    try:
        workers_ref = db.collection('trabajadores')
        new_user_ref = workers_ref.document()
        new_user_id = new_user_ref.id
        
        new_user_ref.set({
            'estadoTrabajador': 'incompleto',
            'created_at': firestore.SERVER_TIMESTAMP,
            'perfil': {
                'name': f"{nombres} {apellidos}",
                'createdAt': firestore.SERVER_TIMESTAMP
            }
        }, merge=True)
        
        return {
            "status": "success",
            "user_id": new_user_id,
            "message": "Datos registrados. Por favor suba sus documentos."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando datos: {str(e)}")

async def process_kyc_update(
    user_id: str, 
    ci_front: bytes, 
    ci_back: bytes, 
    license_front: bytes = None,
    license_back: bytes = None
) -> dict:
    try:
        workers_ref = db.collection('trabajadores')
        worker_doc = workers_ref.document(user_id)
        
        doc_snapshot = worker_doc.get()
        if not doc_snapshot.exists:
            raise HTTPException(status_code=404, detail="Usuario no encontrado en el sistema.")

        bucket = storage.bucket()

        def upload_document_image(image_bytes: bytes, doc_type: str) -> str:
            carpeta_principal = "ci" if "ci" in doc_type else "licencia"
            nombre_archivo = f"trabajadores/{user_id}/documentosPersonales/{carpeta_principal}/{doc_type}_{uuid.uuid4().hex[:8]}.jpg"
            blob = bucket.blob(nombre_archivo)
            blob.upload_from_string(image_bytes, content_type='image/jpeg')
            blob.make_public()
            return blob.public_url

        async def upload_if_exists(image_bytes: bytes, doc_type: str):
            if not image_bytes:
                return None
            return await asyncio.to_thread(upload_document_image, image_bytes, doc_type)

        url_ci_front, url_ci_back, url_license_front, url_license_back = await asyncio.gather(
            upload_if_exists(ci_front, "ci_anverso"),
            upload_if_exists(ci_back, "ci_reverso"),
            upload_if_exists(license_front, "licencia_anverso"),
            upload_if_exists(license_back, "licencia_reverso")
        )

        documentos = {
            'ci': {
                'frontalUrl': url_ci_front,
                'reversoUrl': url_ci_back,
                'estado': 'en_revision',
                'mensajeRevision': '',
                'uploadedAt': firestore.SERVER_TIMESTAMP
            }
        }

        if url_license_front or url_license_back:
            documentos['licencia'] = {
                'frontalUrl': url_license_front,
                'reversoUrl': url_license_back,
                'estado': 'en_revision',
                'mensajeRevision': '',
                'uploadedAt': firestore.SERVER_TIMESTAMP
            }

        worker_doc.update({
            'documentosPersonales': documentos
        })

        return {
            "status": "success",
            "message": "Documentos guardados exitosamente. Proceda al escaneo facial.",
            "user_id": user_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando documentos: {str(e)}")

async def process_initial_face_login(photo_bytes: bytes) -> dict:
    try:
        # Analizamos la selfie
        analisis = await asyncio.to_thread(_analizar_rostro_completo, photo_bytes)
        
        if not analisis["is_real"]:
            db.collection('fraud_attempts').add({
                'action': 'initial_face_login', 
                'liveness_score': analisis["liveness_score"], 
                'timestamp': firestore.SERVER_TIMESTAMP
            })
            raise HTTPException(status_code=403, detail="Alerta de Seguridad: Prueba de vida fallida (Posible Spoofing).")

        live_embedding = analisis["embedding"]

        # Buscar todos los trabajadores que están activos (y que tienen hash)
        workers_ref = db.collection('trabajadores')
        query = workers_ref.where(filter=firestore.FieldFilter("estadoTrabajador", "==", "activo"))
        active_workers = query.stream()

        mejor_similitud = 0.0
        mejor_worker = None

        for worker in active_workers:
            doc_data = worker.to_dict()
            saved_embedding = doc_data.get('face_embedding')
            
            if saved_embedding:
                similitud = _calcular_similitud(live_embedding, saved_embedding)
                if similitud > mejor_similitud:
                    mejor_similitud = similitud
                    mejor_worker = worker

        if mejor_worker and mejor_similitud >= SIMILARITY_THRESHOLD:
            doc_data = mejor_worker.to_dict()
            mejor_worker.reference.update({
                'ultimaValidacionRostroAt': firestore.SERVER_TIMESTAMP,
            })
            return {
                "status": "success",
                "user_id": mejor_worker.id,
                "action": "logged_in",
                "message": f"Bienvenido de vuelta, {doc_data.get('perfil', {}).get('name', 'Usuario')}.",
                "liveness_score": round(analisis["liveness_score"], 4)
            }

        # Si recorre todos los trabajadores activos y ninguno hace match o supera el umbral
        return {
            "status": "not_found",
            "message": "Rostro no registrado en el sistema. Redirigiendo a registro."
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno en login: {str(e)}")

async def _verificar_rostro_documento(worker_doc, documentos, tipo_doc, nombre_legible, live_embedding, umbral):
    doc = documentos.get(tipo_doc, {})
    url = doc.get('frontalUrl')
    estado = doc.get('estado')

    if not url or estado == 'aprobado':
        return

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            doc_bytes = response.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error descargando {nombre_legible} para verificación biométrica. {e}")

    try:
        analisis_doc = await asyncio.to_thread(_analizar_rostro_completo, doc_bytes)
        similitud = _calcular_similitud(live_embedding, analisis_doc["embedding"])
        
        if similitud < umbral:
            worker_doc.update({
                f'documentosPersonales.{tipo_doc}.estado': 'rechazado',
                f'documentosPersonales.{tipo_doc}.mensajeRevision': f'El rostro en la selfie no coincide con el de {nombre_legible}.',
                f'documentosPersonales.{tipo_doc}.revisadoAt': firestore.SERVER_TIMESTAMP
            })
            raise HTTPException(status_code=403, detail=f"El rostro capturado no coincide con el de {nombre_legible}.")

        worker_doc.update({
            f'documentosPersonales.{tipo_doc}.estado': 'aprobado',
            f'documentosPersonales.{tipo_doc}.mensajeRevision': '',
            f'documentosPersonales.{tipo_doc}.revisadoAt': firestore.SERVER_TIMESTAMP,
        })
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Error analizando rostro en {nombre_legible}: {str(ve)}")

async def process_face_registration(user_id: str, photo_bytes: bytes) -> dict:
    try:
        # Se requiere importar _analizar_rostro_completo en el entorno principal
        analisis = await asyncio.to_thread(_analizar_rostro_completo, photo_bytes)
        
        if not analisis["is_real"]:
            db.collection('fraud_attempts').add({
                'user_id': user_id, 
                'action': 'smart_auth_registration', 
                'liveness_score': analisis["liveness_score"], 
                'timestamp': firestore.SERVER_TIMESTAMP
            })
            raise HTTPException(status_code=403, detail="Alerta de Seguridad: Prueba de vida fallida (Posible Spoofing).")

        workers_ref = db.collection('trabajadores')
        worker_doc = workers_ref.document(user_id)
        
        doc_snapshot = worker_doc.get()
        if not doc_snapshot.exists:
            raise HTTPException(status_code=404, detail="Usuario no encontrado.")
            
        doc_data = doc_snapshot.to_dict() or {}
        
        # 1. Si ya tiene face_embedding guardado (Validación de 24 horas o reingreso)
        saved_embedding = doc_data.get('face_embedding')
        UMBRAL_SIMILITUD = 0.50 

        if saved_embedding:
            similitud = _calcular_similitud(analisis["embedding"], saved_embedding)
            if similitud < UMBRAL_SIMILITUD:
                raise HTTPException(status_code=403, detail="Error: El rostro capturado no coincide con el titular de la cuenta.")
            
            worker_doc.update({
                'ultimaValidacionRostroAt': firestore.SERVER_TIMESTAMP,
                'updated_at': firestore.SERVER_TIMESTAMP
            })
            return {
                "status": "success",
                "action": "verified",
                "message": "Verificación biométrica exitosa. Bienvenido de vuelta.",
                "verified": True,
                "liveness_score": round(analisis["liveness_score"], 4)
            }

        # 2. Si NO tiene face_embedding (Registro por primera vez)
        documentos = doc_data.get('documentosPersonales', {})
        
        await _verificar_rostro_documento(worker_doc, documentos, 'ci', 'el carnet', analisis["embedding"], UMBRAL_SIMILITUD)
        await _verificar_rostro_documento(worker_doc, documentos, 'licencia', 'la licencia', analisis["embedding"], UMBRAL_SIMILITUD)

        # 3. Si todo sale bien (o ya estaban aprobados), guardamos la foto y el embedding inicial
        bucket = storage.bucket()
        nombre_archivo = f"fotos_perfil/{user_id}_{uuid.uuid4().hex[:8]}.jpg"
        blob = bucket.blob(nombre_archivo)
        blob.upload_from_string(photo_bytes, content_type='image/jpeg')
        blob.make_public() 
        photo_url = blob.public_url

        worker_doc.update({
            'face_embedding': analisis["embedding"],
            'perfil.photoUrl': photo_url,
            'estadoTrabajador': 'activo',
            'ultimaValidacionRostroAt': firestore.SERVER_TIMESTAMP,
            'updated_at': firestore.SERVER_TIMESTAMP
        })
        
        return {
            "status": "success",
            "action": "registered_and_verified",
            "message": "Registro y validación completados exitosamente. Bienvenido.",
            "verified": True,
            "liveness_score": round(analisis["liveness_score"], 4)
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")
