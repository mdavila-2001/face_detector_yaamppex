import cv2
import numpy as np
import onnxruntime as ort

def probar_liveness(ruta_imagen, ruta_modelo="models/modelrgb.onnx"):
    print(f"\nAnalizando imagen: {ruta_imagen}")
    
    # 1. Cargar el cerebro Anti-Spoofing
    try:
        session = ort.InferenceSession(ruta_modelo, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"Error cargando el modelo: {e}")
        return

    # Descubrimos dinámicamente qué tamaño de imagen exige el modelo
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    # Generalmente el shape es [1, 3, alto, ancho]. Extraemos alto y ancho.
    alto_requerido = input_shape[2] if isinstance(input_shape[2], int) else 112
    ancho_requerido = input_shape[3] if isinstance(input_shape[3], int) else 112

    # 2. Leer y preparar la imagen
    img = cv2.imread(ruta_imagen)
    if img is None:
        print("No se pudo leer la imagen.")
        return

    # En un sistema real, aquí usaríamos InsightFace para recortar SOLO la cara.
    # Para esta prueba rápida, redimensionamos toda la imagen.
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # El modelo espera RGB
    img_resized = cv2.resize(img_rgb, (ancho_requerido, alto_requerido))
    
    # Normalizar matemáticamente para la red neuronal
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_normalized = np.transpose(img_normalized, (2, 0, 1)) # Cambiar orden a Canales, Alto, Ancho
    img_normalized = np.expand_dims(img_normalized, axis=0) # Añadir dimensión de 'batch'

    # 3. Ejecutar el modelo
    resultado = session.run(None, {input_name: img_normalized})
    
    # La capa Softmax devuelve probabilidades. Asumimos que la clase 1 es "Real".
    probabilidades = resultado[0][0]
    score_real = float(probabilidades[1]) 
    
    print(f"Puntaje de Liveness (Realidad): {score_real * 100:.2f}%")
    if score_real > 0.85:
        print("✅ RESULTADO: Humano Vivo Detectado. Permitiendo paso a reconocimiento facial.")
    else:
        print("❌ ALERTA: Posible ataque detectado (Foto impresa/Pantalla). Acceso denegado.")

# Ejecutamos la prueba con tus fotos
probar_liveness("image_7c0c05.png")
probar_liveness("image_7c0bc7.png")
probar_liveness("image.png")

# Te sugiero que intentes tomarle una foto a tu monitor mostrando una de estas 
# imágenes y la pases por el script como "foto_trampa.jpg" para ver cómo la rechaza.