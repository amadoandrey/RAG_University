from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


# ==========================
# 🔹 Funciones para cargar modelo y tokenizer
# ==========================

def load_tokenizer(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """
    Carga el tokenizador asociado al modelo de lenguaje.

    Args:
        model_name (str, opcional):
            Nombre del modelo de HuggingFace del cual se desea cargar el tokenizador.
            Por defecto: "TinyLlama/TinyLlama-1.1B-Chat-v1.0".

    Returns:
        AutoTokenizer: Objeto que convierte texto en tokens comprensibles por el modelo.
    """
    return AutoTokenizer.from_pretrained(model_name)


def load_model(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """
    Carga el modelo de lenguaje desde HuggingFace Hub.

    Se configura para usar GPU automáticamente (si está disponible),
    con precisión en punto flotante optimizada para memoria y velocidad.

    Args:
        model_name (str, opcional):
            Nombre del modelo en HuggingFace Hub.
            Por defecto: "TinyLlama/TinyLlama-1.1B-Chat-v1.0".

    Returns:
        AutoModelForSeq2SeqLM: Modelo de lenguaje preentrenado listo para generación de texto.
    """
    return AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        device_map="auto",   # Detecta automáticamente si usar CPU o GPU
        torch_dtype="auto"   # Ajusta el tipo de datos según el hardware (float16 en GPU, float32 en CPU)
    )


def create_text_pipeline(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Modelo de HuggingFace a utilizar
    max_new_tokens: int = 300,                               # Máximo de tokens que puede generar la respuesta
    repetition_penalty: float = 1.8,                         # Penalización para evitar repeticiones en la salida
    temperature: float = 0.8,                                # Controla la aleatoriedad: bajo=determinista, alto=creativo
    top_p: float = 0.9                                       # Nucleus sampling: porcentaje de palabras más probables
):
    """
    Crea un pipeline de HuggingFace para generación de texto dentro del flujo RAG.

    Args:
        model_name (str, opcional): Nombre del modelo de HuggingFace a utilizar.
                                    Por defecto: "TinyLlama/TinyLlama-1.1B-Chat-v1.0".
        max_new_tokens (int, opcional): Número máximo de tokens que puede generar el modelo en la salida.
                                        Por defecto: 300.
        repetition_penalty (float, opcional): Penalización para evitar repeticiones innecesarias en el texto generado.
                                              Por defecto: 1.8.
        temperature (float, opcional): Controla la aleatoriedad en la generación de texto.
                                       Valores bajos generan respuestas más determinísticas,
                                       valores altos generan respuestas más creativas.
                                       Por defecto: 0.8.
        top_p (float, opcional): Estrategia de muestreo "nucleus sampling".
                                 Controla la diversidad limitando las palabras candidatas al
                                 porcentaje acumulado `p` más probable.
                                 Por defecto: 0.9.

    Returns:
        pipeline: Objeto `pipeline` de HuggingFace configurado para generación de texto.
    """
    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name)

    return pipeline(
        "text2text-generation",    # Tipo de tarea: generación de texto a texto
        model=model,               # Modelo de lenguaje cargado
        tokenizer=tokenizer,       # Tokenizador correspondiente
        truncation=True,           # Truncar texto largo que exceda el límite
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id  # Usar token de fin de secuencia como relleno
    )
