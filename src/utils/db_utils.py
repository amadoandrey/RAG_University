from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# ==========================
# 🔹 Guardar documentos en Chroma
# ==========================
def save_to_chroma(
    documents,
    persist_dir: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
):
    """
    Guarda documentos en ChromaDB usando embeddings de HuggingFace.
    - documents: lista de Document
    - persist_dir: carpeta donde se guardará la BD
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    vectordb.persist()
    return vectordb


# ==========================
# 🔹 Obtener un retriever desde Chroma
# ==========================
def get_retriever(
    modulo: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    search_type: str = "mmr",
    k: int = 8,
    fetch_k: int = 20,
    lambda_mult: float = 0.7
):
    """
    Carga la base vectorial de un módulo y devuelve un retriever.
    - modulo: nombre del módulo (ej: 'Estudiantes')
    - search_type: tipo de búsqueda ('mmr' o 'similarity')
    - k: número de resultados finales
    - fetch_k: número de documentos a traer antes de filtrar
    - lambda_mult: balance entre diversidad y similitud (solo para MMR)
    """
    persist_dir = f"data/BD/{modulo.lower()}"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    retriever = vectordb.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult}
    )
    return retriever
