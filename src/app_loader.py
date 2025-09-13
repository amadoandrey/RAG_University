import streamlit as st
from pathlib import Path
import os, uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.text_processing import split_texts_general
from utils.db_utils import save_to_chroma

# --- Configuración de la página ---
st.set_page_config(page_title="Cargar documentos RAG por módulo", page_icon="📁")

# --- Logo y cabecera ---
def _find_logo():
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent / "images" / "Logo_poli.jpg",
        Path.cwd() / "images" / "Logo_poli.jpg",
        Path("/content/images/Logo_poli.jpg"),
        Path("/content/drive/MyDrive/images/Logo_poli.jpg"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None

col1, col2 = st.columns([1, 5])
with col1:
    logo_path = _find_logo()
    if logo_path:
        st.image(logo_path, width=100)
with col2:
    st.markdown(
        """
        <div style='padding-top: 15px;'>
            <h3 style='margin-bottom: 0;'>📁 Cargar documentos RAG por módulo</h3>
            <p style='font-size: 18px; margin-top: 2px;'>Universidad Politécnico Grancolombiano</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- Selección del módulo ---
modulo = st.selectbox("Seleccione el módulo al que pertenecen los documentos:", ["Estudiantes", "Profesores", "Administrativos"])
PERSIST_DIR = f"data/BD/{modulo.lower()}"
st.info(f"Los documentos se guardarán en: `{PERSIST_DIR}`")

# --- Fragmentación ---
fragmentacion_opcion = st.radio(
    "¿Cómo deseas dividir los textos?",
    ["Fragmentación por longitud (chunks)", "Fragmentación por estructura (título, artículo, sección)"]
)

chunk_size, chunk_overlap = 1200, 200
if fragmentacion_opcion == "Fragmentación por longitud (chunks)":
    chunk_size = st.slider("Tamaño de fragmento", 500, 2000, 1200, step=100)
    chunk_overlap = st.slider("Solapamiento entre fragmentos", 0, 500, 200, step=50)

# --- Subida de archivos ---
uploaded_files = st.file_uploader("Sube uno o varios archivos PDF", type=["pdf"], accept_multiple_files=True)

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
            st.error(f"❌ Error procesando '{uploaded_file.name}': {e}")

        progress_bar.progress((i + 1) / total_files)

    if all_chunks:
        save_to_chroma(all_chunks, PERSIST_DIR)
        st.session_state["fragmentos_generados"] = all_chunks
        st.success(f"✅ Todos los documentos se han guardado en `{PERSIST_DIR}`")

if "fragmentos_generados" in st.session_state:
    if st.checkbox("🔍 Ver primeros fragmentos generados", key="ver_fragmentos_checkbox"):
        for i, chunk in enumerate(st.session_state["fragmentos_generados"][:5]):
            st.markdown(f"**Fragmento {i+1}:**")
            st.code(chunk.page_content[:500])
