import firebase_admin
from firebase_admin import credentials, firestore
import os

def get_db():
    if not firebase_admin._apps:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cred_path = os.path.join(base_dir, "firebase_credentials.json")
        
        if not os.path.exists(cred_path):
            raise RuntimeError(f"Falta el archivo de credenciales en: {cred_path}")
            
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
    
    return firestore.client()

db = get_db()