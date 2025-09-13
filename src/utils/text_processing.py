import re
from langchain.docstore.document import Document

# ==========================
# 🔹 Funciones para fragmentación de documentos
# ==========================
def split_texts_general(documents):
    """
    Divide los textos por estructura (TÍTULO, CAPÍTULO, SECCIÓN, ARTÍCULO).
    Retorna una lista de Document con fragmentos limpios.
    """
    all_chunks = []
    pattern = (
        r"((?:T[ÍI]TULO|CAP[ÍI]TULO|SECCI[ÓO]N|ART[ÍI]CULO)\s+[\w\d\-]+[\s\S]*?)"
        r"(?=\n(?:T[ÍI]TULO|CAP[ÍI]TULO|SECCI[ÓO]N|ART[ÍI]CULO)\s+[\w\d\-]+|\Z)"
    )
    for doc in documents:
        text = doc.page_content
        metadata = doc.metadata
        matches = re.findall(pattern, text, re.IGNORECASE)

        if not matches:
            all_chunks.append(Document(page_content=text.strip(), metadata=metadata))
        else:
            for match in matches:
                if len(match.strip()) > 50:  # Evita fragmentos demasiado pequeños
                    all_chunks.append(Document(page_content=match.strip(), metadata=metadata))
    return all_chunks


# ==========================
# 🔹 Funciones para limpiar y procesar respuestas de LLM
# ==========================
def process_llm_response(response: str) -> str:
    """
    Limpia la salida de un modelo de lenguaje.
    - Elimina espacios extra
    - Devuelve un mensaje estándar si está vacío
    """
    if response and response.strip():
        return response.strip()
    return "Lo siento, pero no tengo información suficiente para responder a esa pregunta."


def obtener_respuesta_desde_respuesta_tag(texto: str) -> str:
    """
    Extrae solo el contenido después de la etiqueta 'Respuesta:' si existe.
    """
    if "Respuesta:" in texto:
        return texto.split("Respuesta:")[-1].strip()
    return texto.strip()
