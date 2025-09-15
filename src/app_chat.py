import streamlit as st
from pathlib import Path

# === Importar funciones auxiliares ===
from utils.db_utils import get_retriever
from utils.text_processing import process_llm_response, obtener_respuesta_desde_respuesta_tag
from utils.evaluation import evaluar_respuesta, limpiar_respuesta
from utils.model_utils import create_text_pipeline

# === Configuración de la página ===
st.set_page_config(page_title="Consultas Chatbot RAG", page_icon="🤖")


# === Función auxiliar: búsqueda del logo ===
def _find_logo() -> str | None:
    """
    Busca el logo institucional en varias rutas candidatas del sistema.

    Esto hace que la aplicación sea más robusta en diferentes entornos 
    (local, Colab, Google Drive). Recorre posibles rutas hasta encontrar el archivo.

    Returns:
        str | None:
            - Ruta completa del logo si se encuentra.
            - None si no existe en ninguna de las rutas.
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent / "images" / "Logo_poli.jpg",     # Repositorio local: /images
        Path.cwd() / "images" / "Logo_poli.jpg",             # Directorio actual
        Path("/content/images/Logo_poli.jpg"),               # Google Colab
        Path("/content/drive/MyDrive/images/Logo_poli.jpg"), # Google Drive en Colab
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


# === Cabecera con logo y título ===
col1, col2 = st.columns([1, 6])
with col1:
    logo_path = _find_logo()
    if logo_path:
        st.image(logo_path, width=90)
    else:
        st.markdown("### 📘")  # Fallback si no se encuentra el logo
with col2:
    st.markdown(
        """
        <div style="margin-top: 6px;">
            <span style="font-size: 1.0em; color:#3B486A; font-weight: 600; line-height:1.15;">
                Universidad Politécnico<br>Grancolombiano
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Línea divisoria
st.markdown(
    "<hr style='margin:0.6em 0 1.2em 0; border:1px solid #e5e7eb;'>",
    unsafe_allow_html=True,
)

# Subtítulo descriptivo
st.markdown(
    """
    <div style='text-align:left;'>
        <span style='font-size:2.0em; font-weight:600; color:#232949;'>
            🤖 Chatbot RAG - Universidad Politécnico Grancolombiano
        </span>
        <p style='font-size:1.1em; color:#444; margin-top:0.4em;'>
            Haz preguntas sobre los documentos cargados previamente en la base de datos.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# === Estado de sesión ===
# Variables de control que guardan historial, última respuesta y estado de consulta.
if "historial" not in st.session_state:
    st.session_state["historial"] = []  # [(pregunta, respuesta, fuentes)]
if "ultima_respuesta" not in st.session_state:
    st.session_state["ultima_respuesta"] = None
if "consulta_realizada" not in st.session_state:
    st.session_state["consulta_realizada"] = False


# === Selección del módulo ===
# El usuario debe escoger el módulo (Estudiantes, Profesores, Administrativos).
modulo = st.selectbox(
    "Seleccione el módulo:",
    ["Seleccione un módulo...", "Estudiantes", "Profesores", "Administrativos"],
)

if modulo == "Seleccione un módulo...":
    st.warning("⚠️ Debes seleccionar un módulo antes de continuar.")
    st.stop()

st.info(f"📚 Módulo seleccionado: **{modulo}**")


# === Carga del modelo y retriever ===
# Se inicializa el modelo de lenguaje (llm) y el recuperador de documentos (retriever).
llm = create_text_pipeline()       # Modelo de lenguaje HuggingFace adaptado a LangChain
retriever = get_retriever(modulo)  # Recuperador basado en la base vectorial del módulo


# === Formulario de consulta ===
# El usuario ingresa una pregunta y se procesa al enviar el formulario.
with st.form("consulta_form", clear_on_submit=True):
    query = st.text_input(
        "Pregunta (en español):",
        key="pregunta_input",
        placeholder="Ejemplo: ¿Cuál es la misión institucional?",
    )
    enviar = st.form_submit_button("Consultar")

if enviar and query.strip():
    st.info("Consultando al modelo RAG...")
    try:
        # Recupera documentos relevantes y genera una respuesta
        docs = retriever.get_relevant_documents(query)
        respuesta = llm(query)

        # Procesa la respuesta para dejarla limpia y clara
        texto_respuesta = process_llm_response(respuesta[0]["generated_text"])
        texto_respuesta = obtener_respuesta_desde_respuesta_tag(texto_respuesta)

        if not docs:
            st.warning("Lo siento, no encontré información suficiente en los documentos.")
            st.session_state["ultima_respuesta"] = None
        else:
            st.session_state["historial"].append((query, texto_respuesta, docs))
            st.session_state["ultima_respuesta"] = (query, texto_respuesta, docs)
            st.success("✅ Consulta realizada con éxito")

    except Exception as e:
        st.error(f"Error en la consulta: {e}")
        st.session_state["ultima_respuesta"] = None


# === Mostrar respuesta principal ===
if st.session_state["ultima_respuesta"]:
    query, respuesta_txt, fuentes = st.session_state["ultima_respuesta"]
    st.markdown(f"**Pregunta:** `{query}`")
    st.markdown(f"**Respuesta:**\n\n{respuesta_txt}")

    if fuentes:
        with st.expander("📂 Documentos fuente consultados"):
            for i, doc in enumerate(fuentes, 1):
                preview = doc.page_content[:400].replace("\n", " ").strip() + "..."
                st.markdown(f"**{i}. {doc.metadata.get('source', 'Documento')}**\n\n*Vista previa:* {preview}\n")


# === Mostrar historial de preguntas y respuestas ===
if st.session_state["historial"]:
    st.markdown("### 📝 Historial de preguntas y respuestas")
    for idx, (pregunta, respuesta, fuentes) in enumerate(st.session_state["historial"][::-1], 1):
        with st.expander(f"Pregunta #{len(st.session_state['historial'])-idx+1}: {pregunta}"):
            st.markdown(f"**Respuesta:** {respuesta}")


# === Métricas automáticas de evaluación ===
# Permite comparar la respuesta generada con una referencia usando métricas BLEU, ROUGE y METEOR.
mostrar_metricas = st.checkbox("📊 Mostrar métricas automáticas de evaluación")

if mostrar_metricas and st.session_state.get("ultima_respuesta"):
    query, texto_respuesta_, fuentes = st.session_state["ultima_respuesta"]
    texto_respuesta = limpiar_respuesta(texto_respuesta_)
    resultados = evaluar_respuesta(query, texto_respuesta)

    if resultados:
        col1, col2, col3 = st.columns(3)
        col1.metric("BLEU", resultados["BLEU"])
        col2.metric("ROUGE-L", resultados["ROUGE-L"])
        col3.metric("METEOR", resultados["METEOR"])

        st.markdown("**Referencia esperada:**")
        st.info(resultados["respuesta_referencia"])
    else:
        st.warning("No se encontró una respuesta de referencia para esta pregunta en el archivo CSV.")
