import re
from langchain.docstore.document import Document


# ==========================
# 🔹 Funciones para fragmentación de documentos
# ==========================
def split_texts_general(documents):
    """
    Divide los textos en fragmentos según su estructura formal
    (TÍTULO, CAPÍTULO, SECCIÓN, ARTÍCULO).

    Args:
        documents (list[Document]): Lista de documentos de entrada, 
                                    cada uno con texto y metadatos.

    Returns:
        list[Document]: Lista de objetos `Document` con fragmentos extraídos
                        y metadatos preservados.
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
            # Si no hay coincidencias, guarda el texto completo como un solo fragmento
            all_chunks.append(Document(page_content=text.strip(), metadata=metadata))
        else:
            for match in matches:
                # Se descartan fragmentos demasiado pequeños para ser útiles
                if len(match.strip()) > 50:
                    all_chunks.append(Document(page_content=match.strip(), metadata=metadata))
    return all_chunks


# ==========================
# 🔹 Funciones para limpiar y procesar respuestas de LLM
# ==========================
def process_llm_response(response: str) -> str:
    """
    Limpia y valida la salida de un modelo de lenguaje.

    Args:
        response (str): Texto generado por el modelo.

    Returns:
        str: Texto limpio y sin espacios extra. 
             Si la respuesta está vacía, devuelve un mensaje estándar 
             indicando que no hay información suficiente.
    """
    if response and response.strip():
        return response.strip()
    return "Lo siento, pero no tengo información suficiente para responder a esa pregunta."


def obtener_respuesta_desde_respuesta_tag(texto: str) -> str:
    """
    Extrae el contenido relevante después de la etiqueta 'Respuesta:'.

    Args:
        texto (str): Texto completo generado por el modelo.

    Returns:
        str: Solo la parte del texto posterior a 'Respuesta:'.
             Si la etiqueta no existe, devuelve el texto original limpio.
    """
    if "Respuesta:" in texto:
        return texto.split("Respuesta:")[-1].strip()
    return texto.strip()
