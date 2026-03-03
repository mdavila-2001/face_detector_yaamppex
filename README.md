# Sistema de Autenticación Facial y Anti-Spoofing (Liveness)

Este es un microservicio backend de alto rendimiento desarrollado en **FastAPI** para realizar validación de identidad y pruebas de vida (Liveness) en tiempo real. Está diseñado específicamente para integrarse en flujos de validación de conductores o usuarios críticos en plataformas que exigen máxima seguridad.

## Arquitectura de Modelos de IA
El sistema utiliza una arquitectura híbrida de Inteligencia Artificial que balancea la precisión comercial con el cumplimiento de licencias Open Source (MIT):

1. **Detección Facial y Landmarks (InsightFace `buffalo_l`):** Utilizado exclusivamente para ubicar dónde está el rostro en la foto y encontrar sus 5 puntos clave (ojos, nariz, comisuras).
2. **Prueba de Liveness / Anti-Spoofing (MiniFASNet `modelrgb.onnx`):** Un modelo clasificador entrenado para distinguir entre un rostro tridimensional (persona real) y ataques de suplantación bidimensionales (fotos impresas, pantallas de tablets o teléfonos).
3. **Reconocimiento de Identidad (ArcFace ResNet100 `arcfaceresnet100-8.onnx`):** Modelo bajo **Licencia MIT** (apto para uso comercial sin restricciones). Transforma el rostro matemáticamente alineado en un vector único de 512 dimensiones (Embedding), que luego es comparado mediante Similitud de Coseno.

---

## Flujo de Procesamiento Optimizado
Todo el flujo de análisis está empaquetado y optimizado en CPU para ejecutarse en nanosegundos (aunque soporta `CUDAExecutionProvider` para GPUs si el servidor lo tiene disponible).

1. **Recepción:** Recibe la foto en bytes e inicializa la decodificación.
2. **Identificación Temprana (Fail-Fast):** Detecta la caja delimitadora (Bounding Box) del rostro. Si hay 0 o más de 1 rostro, aborta instantáneamente para ahorrar recursos computacionales.
3. **Análisis de Contexto (Liveness):** Realiza un recorte *expandido al 150%* alrededor de la cabeza para capturar el fondo, lo redimensiona a 112x112 y determina si la persona está viva o no. Si el modelo arroja una probabilidad menor al `85%`, se rechaza al instante y se lanza una **Alerta de Seguridad**.
4. **Validación de Identidad (Solo si es persona real):** Usando los Puntos de Referencia, recorta de manera matemática y precisa la cara pura (alineación `norm_crop`), la procesa en ArcFace y obtiene el Vector Crudo.
5. **Normalización L2 Matemáticamente Pura:** Crucial para buscar la similitud, el vector se divide por su propia norma algebraica, asegurando escalabilidad y mediciones exactas en bases de datos masivas.

---

## Endpoints de la API

### 1. Smart Auth / Login Biométrico (`POST /auth`)
Integra tanto el Registro Inicial (Lazy Registration) como la Validación Recurrente en un solo endpoint inteligente.
Busca al trabajador en cascada por su CI o Pasaporte. Si no existe, lo crea asignándole un estado seguro `'incompleto'` hasta que envíe sus documentos. Si existe y la biometría coincide, valida su ingreso a operaciones regulares.

**Input Form Data:**
- `document_number`: (String) Número de Cédula o Pasaporte.
- `live_photo`: (File) Fotografía biométrica en vivo recién tomada para evaluar su liveness e identidad.

**Comportamientos y Respuestas (200 OK):**
La app en Flutter decide la redirección de pantalla basándose en la variable `action` que devuelve el servidor:
- `"action": "created_pending_kyc"`: Perfil borrador creado. Debe enviarse al conductor al formulario KYC Onboarding de inmediato.
- `"action": "verified"`: Verificación matemática exitosa. El usuario accede a la pestaña principal.
- `"action": "biometrics_updated"`: Primera foto en vivo sincronizada a la perfección con el perfil administrativo existente.

### 2. Actualización de Documentos KYC (`POST /update_kyc`)
Último paso determinante donde el trabajador sube su set de documentos personales para formalizar su inscripción, desbloqueando su estado borrador ('incompleto') al estatus `'activo'`. Utiliza procesadores subyacentes múltiples (`asyncio.gather`) permitiendo que de dos a cuatro fotos pesadas se envíen a Google Cloud simultáneamente.

**Input Form Data:**
- `user_id`, `nombres`, `apellidos`: Datos base en strings.
- `ci_front`, `ci_back`: (Archivos Obligatorios) Rostro y contracara del Carnet.
- `passport_front`, `passport_back`: (Archivos Opcionales) Rostro y contracara legal del Pasaporte.

---

## Puntos de Integración con Firebase
El microservicio interactúa ininterrumpidamente con las infraestructuras de Base de datos y Almacenamiento:
* **Colección `trabajadores/{user_id}`**: Corazón del sistema del conductor. Almacena en estado unificado (`perfil`, `documentos`) sus métricas en línea y su huella irreplicable global (`face_embedding`).
* **Colección `fraud_attempts/{auto_id}`**: Auditorías Anti-Spoofing. Si un intento choca estrepitosamente en el sistema (ej. fotografía tomada a la pantalla de un monitor), aparta las credenciales asaltantes resguardando el ecosistema.
* **Storage `fotos_perfil/` y `documentos_kyc/`**: Gestores fotográficos organizados vía UUIDs, listos para arrojar firmas URL blindadas a perfiles sin temor a colisión de nomenclatura.

---

## Mantenimiento y Dependencias del Entorno
Si tus clientes necesitan saber el nivel de estabilidad tecnológica, este proyecto fue anclado (*pinned*) a versiones maduras y estables de las librerías matemáticas más pesadas de Python.

Cualquier esfuerzo de "Mantenimiento" o "Actualización" futura debe respetar la matriz de compatibilidades de estos paquetes, declarados en el archivo `requirements.txt`:

| Dependencia | Versión Fija | Resumen de Responsabilidad |
| :--- | :--- | :--- |
| **`fastapi`** | `0.109.2` | Manejo del servidor web asíncrono y los endpoints de API. |
| **`uvicorn`** | `0.27.1` | El servidor *daemon* subyacente que ejecuta FastAPI. |
| **`onnxruntime`** | `1.16.3` | El motor *C++* altamente optimizado que ejecuta las redes neuronales Liveness y ArcFace sin requerir pesados frameworks como PyTorch/TensorFlow. |
| **`insightface`** | `0.7.3` | Únicamente usado para su algoritmo `2d106det` (encontrar rostros a ultra-rapidez) y utilidades de alineamiento matemático de imágenes. |
| **`opencv-python-headless`** | `4.9.0.80` | Procesamiento interno ultra-veloz de imágenes (recortes, redimensionamiento, colores). Se usa la versión `headless` porque no se requiere pintar ventanas pop-up en el servidor. |
| **`firebase-admin`** | `6.4.0` | Conexión Oficial Servidor-a-Servidor para Cloud Firestore. |
| **`numpy`** | `1.26.4` | El corazón matemático puro para la manipulación y cálculos de los tensores y álgebra lineal. |
| **`scipy`** | `1.12.0` | Complemento matemático optimizado para distancias. |
| **`python-multipart`**| `0.0.9` | Analizador para poder recibir archivos binarios por HTTP `Form-Data`. |

### Cómo Instalar en un Entorno Nuevo
Si tus clientes migran a otro servidor, los pasos son simples:
1. Instalar Python 3.12 o superior.
2. Crear un entorno virtual (`python -m venv .env` y `source .env/bin/activate`).
3. Instalar las dependencias exactas: `pip install -r requirements.txt`.
4. Descargar los modelos en la carpeta `models/`.
5. Ejecutar la app: `uvicorn app.main:app --host 0.0.0.0 --port 8000`.