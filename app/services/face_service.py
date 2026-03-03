import cv2
import numpy as np
import asyncio
import onnxruntime as ort
from fastapi import HTTPException
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop
from app.db.firebase import db
from google.cloud import firestore
import uuid
from firebase_admin import storage

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

async def process_smart_auth(document_number: str, photo_bytes: bytes) -> dict:
    try:
        # 1. Analizar la foto en vivo (Liveness + Extracción del Vector)
        analisis = await asyncio.to_thread(_analizar_rostro_completo, photo_bytes)
        
        # Si es una foto a una pantalla/papel, bloqueamos al instante
        if not analisis["is_real"]:
            db.collection('fraud_attempts').add({
                'document_number': document_number, 
                'action': 'smart_auth', 
                'liveness_score': analisis["liveness_score"], 
                'timestamp': firestore.SERVER_TIMESTAMP
            })
            raise HTTPException(status_code=403, detail="Alerta de Seguridad: Prueba de vida fallida (Posible Spoofing).")

        workers_ref = db.collection('trabajadores')
        user_doc = None
        user_id = None
        
        # 2. Búsqueda en cascada: Primero por CI
        query_ci = workers_ref.where('ci', '==', document_number).get()
        
        if len(query_ci) > 0:
            user_doc = query_ci[0]
            user_id = user_doc.id
        else:
            # 3. Si no existe por CI, buscamos por Pasaporte
            query_pasaporte = workers_ref.where('pasaporte', '==', document_number).get()
            if len(query_pasaporte) > 0:
                user_doc = query_pasaporte[0]
                user_id = user_doc.id

        # 4. LÓGICA DE DECISIÓN (Crear vs Verificar)
        if not user_doc:
            # ESCENARIO A: EL USUARIO NO EXISTE -> LO CREAMOS EN BORRADOR (Lazy Registration)
            
            new_user_ref = workers_ref.document() # Generamos el documento vacío
            new_user_id = new_user_ref.id # ¡Capturamos el ID generado!
            
            # --- Subir foto a Firebase Storage ---
            bucket = storage.bucket()
            nombre_archivo = f"fotos_perfil/{document_number}_{uuid.uuid4().hex[:8]}.jpg"
            blob = bucket.blob(nombre_archivo)
            blob.upload_from_string(photo_bytes, content_type='image/jpeg')
            blob.make_public() 
            photo_url = blob.public_url
            # --------------------------------------------

            new_user_ref.set({
                'ci': document_number,
                'estadoTrabajador': 'incompleto', # 🔒 CANDADO: No puede operar aún
                'face_embedding': analisis["embedding"],
                'created_at': firestore.SERVER_TIMESTAMP,
                'perfil': {
                    'photoUrl': photo_url, 
                    'createdAt': firestore.SERVER_TIMESTAMP
                }
            }, merge=True)
            
            return {
                "status": "success",
                "action": "created_pending_kyc",
                "user_id": new_user_id,
                "message": "Perfil base creado. Por favor complete sus datos y documentos.",
                "verified": True,
                "similarity_score": 1.0,
                "liveness_score": round(analisis["liveness_score"], 4)
            }
            
        else:
            # ESCENARIO B: EL USUARIO SÍ EXISTE -> VERIFICAMOS EL ROSTRO
            user_data = user_doc.to_dict()
            saved_embedding = user_data.get('face_embedding')
            
            # Sub-escenario B1: Existe en la BD pero no tiene biometría previa (Sincronización)
            if not saved_embedding:
                workers_ref.document(user_id).set({
                    'face_embedding': analisis["embedding"],
                    'updated_at': firestore.SERVER_TIMESTAMP
                }, merge=True)
                
                return {
                    "status": "success",
                    "action": "biometrics_updated",
                    "message": "Usuario encontrado, pero no tenía rostro registrado. Biometría guardada exitosamente.",
                    "verified": True,
                    "similarity_score": 1.0,
                    "liveness_score": round(analisis["liveness_score"], 4)
                }
                
            # Sub-escenario B2: Ya tiene biometría -> Hacemos el Match Matemático
            similarity = _calcular_similitud(saved_embedding, analisis["embedding"])
            is_verified = similarity >= SIMILARITY_THRESHOLD
            
            if is_verified:
                # Verificamos si completó su registro
                estado = user_data.get('estadoTrabajador', 'incompleto')
                if estado == 'incompleto':
                    return {
                        "status": "success",
                        "action": "created_pending_kyc", # Lo mandamos a KYC
                        "user_id": user_id,
                        "message": "Bienvenido de vuelta. Por favor, finaliza tu registro completando tus datos personales.",
                        "verified": True,
                        "similarity_score": round(similarity, 4),
                        "liveness_score": round(analisis["liveness_score"], 4)
                    }

                # Si ya está activo:
                workers_ref.document(user_id).set({
                    'last_verified': firestore.SERVER_TIMESTAMP,
                    'status': 'active'
                }, merge=True)
                mensaje = "Verificación exitosa. El rostro coincide con el documento."
            else:
                # 🛑 ¡Suplantador de identidad atrapado de flagrancia matemática!
                mensaje = "Acceso Denegado. El rostro no coincide con el dueño de este documento."
                
            return {
                "status": "success" if is_verified else "failed",
                "action": "verified",
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

async def process_kyc_update(
    user_id: str, 
    nombres: str, 
    apellidos: str, 
    ci_front: bytes, 
    ci_back: bytes, 
    passport_front: bytes = None,
    passport_back: bytes = None
) -> dict:
    try:
        workers_ref = db.collection('trabajadores')
        worker_doc = workers_ref.document(user_id)
        
        doc_snapshot = worker_doc.get()
        if not doc_snapshot.exists:
            raise HTTPException(status_code=404, detail="Usuario no encontrado en el sistema.")

        bucket = storage.bucket()

        def upload_document_image(image_bytes: bytes, doc_type: str) -> str:
            nombre_archivo = f"documentos_kyc/{user_id}/{doc_type}_{uuid.uuid4().hex[:8]}.jpg"
            blob = bucket.blob(nombre_archivo)
            blob.upload_from_string(image_bytes, content_type='image/jpeg')
            blob.make_public()
            return blob.public_url

        async def upload_if_exists(image_bytes: bytes, doc_type: str):
            if not image_bytes:
                return None
            return await asyncio.to_thread(upload_document_image, image_bytes, doc_type)

        url_ci_front, url_ci_back, url_passport_front, url_passport_back = await asyncio.gather(
            upload_if_exists(ci_front, "ci_anverso"),
            upload_if_exists(ci_back, "ci_reverso"),
            upload_if_exists(passport_front, "pasaporte_anverso"),
            upload_if_exists(passport_back, "pasaporte_reverso")
        )

        documentos = {
            'ci_anverso': url_ci_front,
            'ci_reverso': url_ci_back,
            'uploaded_at': firestore.SERVER_TIMESTAMP
        }
        if url_passport_front:
            documentos['pasaporte_anverso'] = url_passport_front
        if url_passport_back:
            documentos['pasaporte_reverso'] = url_passport_back

        # 5. Actualizamos la base de datos (Quitamos el candado)
        worker_doc.update({
            'estadoTrabajador': 'activo', # 🟢 ¡Usuario liberado para trabajar!
            'perfil.name': f"{nombres} {apellidos}", # Usamos dot notation para actualizar dentro del mapa 'perfil'
            'documentos': documentos
        })

        return {
            "status": "success",
            "message": "Registro KYC completado. Los documentos han sido guardados y el perfil está activo."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando KYC: {str(e)}")