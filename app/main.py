from fastapi import FastAPI
from app.api.routes import router
import uvicorn

app = FastAPI(
    title="API de Reconocimiento Facial",
    description="Microservicio con Arquitectura Limpia",
    version="1.0.0"
)

# Conectamos las rutas que creamos
app.include_router(router, prefix="/api/v1")

@app.get("/")
def home():
    return {"mensaje": "API estructurada y en línea 🟢"}

if __name__ == "__main__":
    # Nota: Asegúrate de ejecutar esto desde la raíz del proyecto
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)