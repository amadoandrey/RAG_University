
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import os
import re
import uuid

# --- Configuración de la página ---
st.set_page_config(page_title="Cargar documentos RAG por módulo", page_icon="📁")

# --- Encabezado con logo y título ajustado ---
col1, col2 = st.columns([1, 5])
with col1:
    st.image("/content/drive/MyDrive/Maestría en análitica/Semestre 3/Trabajo de grado/ImágenesUG/Logo_poli.jpg", width=100)
with col2:
    st.markdown(
        """
        <div style='padding-top: 15px;'>
            <h3 style='margin-bottom: 0;'>📁 Cargar documentos RAG por módulo</h3>
            <p style='font-size: 18px; margin-top: 2px;'>Universidad Politécnico GranColombiano</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- Selección del módulo ---
modulo = st.selectbox("Seleccione el módulo al que pertenecen los documentos:", ["Estudiantes", "Profesores", "Administrativos"])
PERSIST_DIR = f"/content/drive/MyDrive/Maestría en análitica/Semestre 3/Trabajo de grado/BD/{modulo.lower()}"
st.info(f"Los documentos se guardarán en: `{PERSIST_DIR}`")

# --- Parámetros de fragmentación ---
fragmentacion_opcion = st.radio(
    "¿Cómo deseas dividir los textos?",
    ["Fragmentación por longitud (chunks)", "Fragmentación por estructura (título, artículo, sección)"]
)

# --- Parámetros de fragmentación por longitud ---
# Si se elige esta opción, el usuario define:
# - chunk_size: tamaño máximo de cada fragmento (500 a 2000, por defecto 1200).
# - chunk_overlap: cantidad de solapamiento entre fragmentos (0 a 500, por defecto 200).
# Estos valores determinan cómo se corta el texto para optimizar la búsqueda semántica.

chunk_size = 1200
chunk_overlap = 200
if fragmentacion_opcion == "Fragmentación por longitud (chunks)":
    chunk_size = st.slider("Tamaño de fragmento", 500, 2000, 1200, step=100)
    chunk_overlap = st.slider("Solapamiento entre fragmentos", 0, 500, 200, step=50)

# --- Función de fragmentación por estructura general ---
def split_texts_general(documents):
    """
    Divide los documentos en fragmentos basados en su estructura (Título, Capítulo, Sección, Artículo).

    Args:
        documents (list[Document]): Lista de documentos cargados, 
                                    cada uno con texto y metadatos.

    Returns:
        list[Document]: Una lista de fragmentos de texto extraídos.  
                        Cada fragmento conserva los metadatos del documento original 
                        (como nombre del archivo o número de página).
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
                if len(match.strip()) > 50:
                    all_chunks.append(Document(page_content=match.strip(), metadata=metadata))
    return all_chunks

# --- Carga de archivos PDF ---
uploaded_files = st.file_uploader("Sube uno o varios archivos PDF", type=["pdf"], accept_multiple_files=True)

# Botón para activar procesamiento
if uploaded_files and st.button("🚀 Cargar documentos"):
    all_chunks = []
    progress_bar = st.progress(0)
    total_files = len(uploaded_files)

    for i, uploaded_file in enumerate(uploaded_files):
        try:
            unique_name = f"{uuid.uuid4().hex}_{uploaded_file.name}"
            file_path = os.path.join("/tmp", unique_name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

            loader = PyPDFLoader(file_path)
            pages = loader.load()
            if not pages:
                st.warning(f"⚠️ El archivo '{uploaded_file.name}' no contiene texto legible.")
                continue

            if fragmentacion_opcion == "Fragmentación por longitud (chunks)":
                splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                chunks = splitter.split_documents(pages)
            else:
                chunks = split_texts_general(pages)

            all_chunks.extend(chunks)
            st.success(f"✅ Procesado: {uploaded_file.name} → {len(chunks)} fragmentos")

        except Exception as e:
            st.error(f" ?Error procesando '{uploaded_file.name}': {e}")

        progress_bar.progress((i + 1) / total_files)

    if all_chunks:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = Chroma.from_documents(documents=all_chunks, embedding=embeddings, persist_directory=PERSIST_DIR )
        st.session_state["fragmentos_generados"] = all_chunks  # Guarda los fragmentos para futura visualización
        vectordb.persist()
        st.info(f"ℹ️ Nombre real de colección: Vector_database_{modulo.lower()}")
        st.success(f"✅ Todos los documentos se han guardado en `{PERSIST_DIR}`")


# Mostrar fragmentos si están en sesión
if "fragmentos_generados" in st.session_state:
    if st.checkbox("🔍 Ver primeros fragmentos generados", key="ver_fragmentos_checkbox"):
        for i, chunk in enumerate(st.session_state["fragmentos_generados"][:5]):
            st.markdown(f"**Fragmento {i+1}:**")
            st.code(chunk.page_content[:500])