from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ==========================
# 🔹 Funciones para cargar modelo y tokenizer
# ==========================

def load_tokenizer(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """
    Carga el tokenizer para el modelo de lenguaje.
    """
    return AutoTokenizer.from_pretrained(model_name)


def load_model(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """
    Carga el modelo de lenguaje (en float16 si hay GPU disponible).
    """
    return AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto"
    )


def create_text_pipeline(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens: int = 300,
    repetition_penalty: float = 1.8,
    temperature: float = 0.8,
    top_p: float = 0.9
):
    """
    Crea un pipeline de generación de texto para consultas RAG.
    """
    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name)

    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id
    )
