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
    Guarda documentos en una base vectorial ChromaDB utilizando embeddings de HuggingFace.

    Args:
        documents (list[Document]): Lista de fragmentos de texto (objetos `Document`).
        persist_dir (str): Carpeta donde se guardará la base de datos vectorial.
        model_name (str, opcional): Modelo de embeddings de HuggingFace a usar.
                                    Por defecto: "sentence-transformers/all-MiniLM-L6-v2".

    Returns:
        Chroma: Objeto de base vectorial Chroma con los documentos almacenados.
    """
    # Crear embeddings a partir del modelo seleccionado
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # Construir la base vectorial con los documentos y embeddings
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    # Guardar en disco para uso posterior
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
    Carga la base vectorial de un módulo y devuelve un objeto retriever para consultas.

    Args:
        modulo (str): Nombre del módulo (ejemplo: 'Estudiantes', 'Profesores').
        model_name (str, opcional): Modelo de embeddings de HuggingFace a usar.
                                    Por defecto: "sentence-transformers/all-MiniLM-L6-v2".
        search_type (str, opcional): Tipo de búsqueda en Chroma. 
                                     - "mmr": Maximal Marginal Relevance (balancea diversidad y relevancia).
                                     - "similarity": búsqueda por similitud directa.
        k (int, opcional): Número de resultados finales a devolver. Por defecto: 8.
        fetch_k (int, opcional): Número de documentos iniciales a recuperar antes de filtrar. Por defecto: 20.
        lambda_mult (float, opcional): Balance entre diversidad (0) y similitud (1) en MMR. Por defecto: 0.7.

    Returns:
        BaseRetriever: Objeto retriever que permite consultar fragmentos relevantes desde la base vectorial.
    """
    # Directorio donde se encuentra la base de datos vectorial del módulo seleccionado
    persist_dir = f"data/BD/{modulo.lower()}"

    # Inicializar embeddings
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # Cargar la base vectorial existente
    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    # Configurar el retriever según el tipo de búsqueda
    retriever = vectordb.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult}
    )

    return retriever
