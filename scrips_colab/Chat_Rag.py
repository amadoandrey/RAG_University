import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from functools import lru_cache
import torch
from huggingface_hub import login

import pandas as pd
import time
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from pathlib import Path

import nltk
# Descargar wordnet automáticamente si no está presente
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')



import torch, gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# --- CONFIGURACIÓN ---
# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Consultas Chatbot RAG", page_icon="🤖")





# --- CABECERA ELEGANTE CON LOGO Y TÍTULO ---
col1, col2 = st.columns([1, 6])
with col1:
    st.image(
        "/content/drive/MyDrive/Maestría en análitica/Semestre 3/Trabajo de grado/ImágenesUG/Logo_poli.jpg",
        width=90
    )
with col2:
    st.markdown("""
        <div style="margin-top: 18px;">
            <span style="font-size: 1.0em; color:#3B486A; font-weight: 600;">
                Universidad Politécnico<br>Grancolombiano
            </span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<hr style="margin:0.5em 0 1.3em 0; border:1px solid #ddd;">
<div style='text-align:left;'>
    <span style='font-size:2.2em; font-weight:600; color:#232949;'>
        🤖 Bienvenido al Chat de la Universidad Politécnico Grancolombiano
    </span>
    <p style='font-size:1.2em; color:#444; margin-top:0.6em; margin-bottom:0.5em;'>
        Haz preguntas sobre los documentos cargados previamente en la base de datos.<br>
        Respuestas claras y confiables para tu comunidad universitaria.
    </p>
</div>
""", unsafe_allow_html=True)

# --- HISTORIAL ---
if 'historial' not in st.session_state:
    st.session_state['historial'] = []  # [(pregunta, respuesta, fuentes)]
if 'ultima_respuesta' not in st.session_state:
    st.session_state['ultima_respuesta'] = None  # (respuesta, fuentes)
if 'consulta_realizada' not in st.session_state:
    st.session_state['consulta_realizada'] = False

# --- Selección del módulo ---
modulo = st.selectbox(
    "Seleccione el módulo que desea consultar:",
    ["Seleccione un módulo...", "Estudiantes", "Profesores", "Administrativos"]
)

if modulo == "Seleccione un módulo...":
    st.warning("⚠️ Debes seleccionar un módulo antes de continuar.")
    st.stop()

PERSIST_DIR = f"/content/drive/MyDrive/Maestría en análitica/Semestre 3/Trabajo de grado/BD/{modulo.lower()}"
REFERENCE_FILE = "/content/drive/MyDrive/Maestría en análitica/Semestre 3/Trabajo de grado/RefeMetricas/referencias_respuestas.csv"
st.info(f"📚 Módulo seleccionado: **{modulo}**")

# --- PARÁMETROS ---
model_name = "meta-llama/Llama-3.2-3B-Instruct"
hf_token = "hf_XXXXXX"  # Si lo necesitas, pon tu token de HuggingFace
login(token=hf_token)

def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

@st.cache_resource
def get_vectordb(PERSIST_DIR):
    embeddings = load_embeddings()
    return Chroma(
        persist_directory=f"{PERSIST_DIR}",
        embedding_function=embeddings
    )

tokenizer = None
model = None


@lru_cache(maxsize=1)
def load_tokenizer():
    global tokenizer
    if tokenizer is None:
        print("🔄 Cargando tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        print("✅ Tokenizer cargado.")
    else:
        print("✅ Tokenizer ya estaba cargado, se reutiliza.")
    return tokenizer

@lru_cache(maxsize=1)
def load_model():
    global model
    if model is None:
        print("🔄 Cargando modelo...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            device_map="auto",
            torch_dtype=torch.float16
        )
        print("✅ Modelo cargado.")
    else:
        print("✅ Modelo ya estaba cargado, se reutiliza.")
    return model

def create_text_pipeline():
    print("🔄 Creando pipeline de generación de texto...")
    tokenizer = load_tokenizer()
    model = load_model()
    return pipeline(
        "text2text-generation", # El tipo de tarea
        model=model, #  El número máximo de tokens generados
        tokenizer=tokenizer, # Tokenizador previamente cargado.
        truncation=True, # Trucamiento para no sobrepasar el límite del modelo.
        max_new_tokens=300, # Número máximo de token de salida del modelo.
        repetition_penalty=1.8,
        temperature=0.8, #controla la aleatoriedad en la generación; valores más bajos hacen la salida más conservadora.
        top_p=0.9, # onsiderando solo el 90% más probable de palabras siguientes.
        pad_token_id=tokenizer.eos_token_id # usa el token de fin de secuencia como relleno cuando sea necesario.

    )



# --- PROMPT CLARO Y CONCISO PARA TinyLlama ---
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Responda específicamente en Español. "
        "Compórtese como un asistente de la Universidad Politécnico Grancolombiano.\n"
        "Solo con base en el contexto proporcionado. No incluya explicaciones adicionales ni contenido repetido..\n\n"
        "{context}\n\n"
        "Pregunta: {question}\n\n"
        "Respuesta:"
    )
)

# --- CARGA VECTOR DB Y MODELO ---
status = st.empty()
status.info("Cargando base vectorial y modelo de lenguaje...")

vectordb = get_vectordb(PERSIST_DIR)
retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 8,
        "fetch_k": 20,
        "lambda_mult": 0.7
    }
)



pipeline_ = create_text_pipeline()
llm = HuggingFacePipeline(pipeline=pipeline_)

qa_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="result" # Esto le dice a la memoria qué clave es la respuesta principal
    )

qa = RetrievalQA.from_chain_type(
    llm=llm,  # Modelo de lenguaje (LLM) que genera las respuestas a partir del contexto proporcionado.
    chain_type="stuff",  # Tipo de cadena que concatena todos los documentos recuperados en un solo texto para el modelo.
    retriever=retriever,  # Componente que busca y devuelve los fragmentos más relevantes desde la base vectorial.
    return_source_documents=True,  # Retorna también los documentos fuente usados para generar la respuesta.
    verbose=True,  # Muestra detalles del proceso durante la ejecución (útil para depuración).
    chain_type_kwargs={"prompt": prompt_template},  # Diccionario que pasa un prompt personalizado al modelo.
    memory = qa_memory,  # Guarda el historial de la conversación para mantener el contexto.
    output_key="result"  # Indica la clave bajo la cual se almacenará la respuesta generada.
     
)

def process_llm_response(llm_response):
    """
    Procesa la respuesta del modelo de lenguaje para extraer y formatear la respuesta relevante.
    """
    new_response = llm_response.get('result', "").strip()  # ✅ Evita errores si 'result' no existe

    # Si la respuesta está vacía, devolver un mensaje de error
    if not new_response:
        return "⚠️ No se recibió una respuesta válida del modelo."

    # 🔹 Buscar el inicio de la respuesta después de "Helpful Answer:"
    start_index = new_response.find("Helpful Answer:")

    if start_index != -1:
        new_response = new_response[start_index + len("Helpful Answer:"):].strip()  # ✅ Extrae solo la respuesta

    # Eliminar cualquier texto adicional no deseado
    new_response = new_response.split("De artículos relevant")[0].strip()
    new_response = new_response.split("Accordingly I was unable")[0].strip()
    new_response = new_response.split("Donnez les réponse")[0].strip()

    # Formatear la respuesta para que sea más legible
    new_response = new_response.replace("- Accreditation", "\n- Accreditation")
    new_response = new_response.replace("You can also check", "\n\nYou can also check")

    return new_response
    

def obtener_respuesta_desde_respuesta_tag(texto_completo):
    """
    Extrae el contenido que sigue después de la palabra 'Respuesta:'.
    Si no se encuentra, devuelve el texto original.
    Al final añade una pregunta para mantener el flujo conversacional.
    """
    clave = "Respuesta:"
    indice = texto_completo.find(clave)

    if indice != -1:
        respuesta = texto_completo[indice + len(clave):].strip()
    else:
        respuesta = texto_completo.strip()

    # Agrega la pregunta estilo chat al final
    respuesta += "\n\n Acá estoy para ayudarte. Digita la siguiente pregunta."

    return respuesta

status.success("Listo para responder preguntas.")

# --- CONSULTA ---

# --- CONSULTA (Bloque Mejorado, SEGURO Y FUNCIONAL) ---

st.markdown("""<hr style="margin:1em 0 1em 0; border:1px solid #ececec;">""", unsafe_allow_html=True)

st.markdown("""
<div style='display: flex; align-items: center; gap:12px; margin-bottom: 0.3em;'>
    <span style='font-size:2.2em; color:#DE2340;'>❓</span>
    <span style='font-size:1.6em; font-weight:600; color:#222;'>Haz tu pregunta sobre los documentos cargados</span>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height: 0.5em;'></div>", unsafe_allow_html=True)

with st.form(key="form_pregunta", clear_on_submit=True):  # <- Esto limpia el input automáticamente
    query = st.text_input(
        "Pregunta (en español):",
        key="pregunta_input",
        placeholder="Ejemplo: ¿Cuál es la misión institucional?"
    )
    enviar = st.form_submit_button("Consultar")
    
    if enviar and query.strip():
        status.info("Consultando al modelo RAG local...")
        try:
            respuesta = qa(query)
            fuentes = respuesta.get("source_documents", [])
            contenido_util = any(doc.page_content.strip() for doc in fuentes)

            if not fuentes or not contenido_util:
                st.session_state['consulta_realizada'] = True
                st.session_state['ultima_respuesta'] = None
                st.warning("Lo siento, pero no tengo información suficiente para responder a esa pregunta. ¿Podrías proporcionar más contexto o detalles? Estoy aquí para ayudarte con cualquier otra pregunta relacionada con el texto proporcionado.")
            else:
                respuesta_completa = process_llm_response(respuesta)
                texto_respuesta = obtener_respuesta_desde_respuesta_tag(respuesta_completa)
                mensaje_no_info = "Lo siento, pero no tengo información suficiente para responder a esa pregunta"
                if mensaje_no_info in texto_respuesta:                  
                  st.warning(mensaje_no_info)
                  st.session_state['consulta_realizada'] = True
                  st.session_state['ultima_respuesta'] = None
                  st.stop()

                query_lower = query.lower() 
                fuente_completa = " ".join([doc.page_content.lower() for doc in fuentes])
                palabras_clave = [p for p in query_lower.split() if len(p) > 0]
                coincidencias = [p for p in palabras_clave if p in fuente_completa]

                if not coincidencias:
                  st.warning("Lo siento, pero no tengo información suficiente para responder a esa pregunta. ¿Podrías proporcionar más contexto o detalles?")
                  st.session_state['ultima_respuesta'] = None
                  st.session_state['consulta_realizada'] = True
                  st.stop()

              

                st.session_state['historial'].append((query, texto_respuesta, fuentes))
                st.session_state['ultima_respuesta'] = (query, texto_respuesta, fuentes)
                st.session_state['consulta_realizada'] = True
                st.success("✅ Consulta realizada con éxito")
            # ¡NO limpies el campo manualmente!
        except Exception as e:
            st.session_state['ultima_respuesta'] = None
            st.session_state['consulta_realizada'] = True
            st.error(f"Error en la consulta: {e}")

st.markdown("""
<div style='font-size:1.05em; color:#326BA2; background-color:#f3f8fc; padding:0.7em 1em; border-radius: 9px; margin-top: 0.8em;'>
    Escribe tu pregunta y haz clic en <b>Consultar</b>.
</div>
""", unsafe_allow_html=True)



# --- MOSTRAR RESPUESTA PRINCIPAL Y DOCUMENTOS FUENTE (expander plano, sin anidar) ---
if st.session_state['consulta_realizada']:
    if st.session_state['ultima_respuesta']:
        query, respuesta_txt, fuentes = st.session_state['ultima_respuesta']
        st.markdown(f"**Pregunta:** `{query}`")
        st.markdown(f"**Respuesta:**\n\n{respuesta_txt}")
    else:
        st.info("*No se encontró información relevante para la consulta.*")


# --- MOSTRAR HISTORIAL, SIN NINGÚN EXPANDER ANIDADO ---
if st.session_state['historial']:
    st.markdown("### 📝 Historial de preguntas y respuestas")
    for idx, (pregunta, respuesta, fuentes) in enumerate(st.session_state['historial'][::-1], 1):
        with st.expander(f"Pregunta #{len(st.session_state['historial'])-idx+1}: {pregunta}", expanded=False):
            st.markdown(f"**Respuesta:** {respuesta}")
            if fuentes:
                st.markdown("**Documentos consultados:**")
                for i, doc in enumerate(fuentes, 1):
                    nombre = doc.metadata.get("source", f"Documento {i}")
                    preview = doc.page_content[:400].replace("\n", " ").strip() + "..."
                    st.markdown(f"**{i}. {nombre}**\n\n*Vista previa:* {preview}\n")
            else:
                st.info("*No se encontraron documentos fuente para esta consulta.*")

@st.cache_data
def cargar_respuestas_referencia():
    try:
        df = pd.read_csv(REFERENCE_FILE)
        df.dropna(subset=["pregunta", "respuesta_referencia"], inplace=True)
        return df
    except Exception as e:
        print(f"Error cargando referencias: {e}")
        return pd.DataFrame(columns=["pregunta", "respuesta_referencia"])

def evaluar_respuesta(pregunta, respuesta_generada):
    df_ref = cargar_respuestas_referencia()
    # Usar la columna "respuesta_referencia"
    respuesta_esperada = df_ref[df_ref.pregunta.str.lower() == pregunta.lower()].respuesta_referencia.values
    if not len(respuesta_esperada):
        return None

    respuesta_esperada = respuesta_esperada[0]
    smoothing = SmoothingFunction().method4
    bleu = sentence_bleu([respuesta_esperada.split()], respuesta_generada.split(), smoothing_function=smoothing)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge = scorer.score(respuesta_esperada, respuesta_generada)["rougeL"].fmeasure
    meteor = meteor_score([respuesta_esperada.split()], respuesta_generada.split())


    return {
        "BLEU": round(bleu, 3),
        "ROUGE-L": round(rouge, 3),
        "METEOR": round(meteor, 3),
        "respuesta_referencia": respuesta_esperada
    }

def limpiar_respuesta(respuesta):
    texto_remover = "Acá estoy para ayudarte. Digita la siguiente pregunta."
    if respuesta.strip().endswith(texto_remover):
        return respuesta.strip()[:-len(texto_remover)].strip()
    else:
        return respuesta.strip()

mostrar_metricas = st.checkbox("📊 Mostrar métricas automáticas de evaluación")

if mostrar_metricas and st.session_state.get("ultima_respuesta"):
    query, texto_respuesta_, fuentes = st.session_state["ultima_respuesta"]

     
    texto_respuesta = limpiar_respuesta(texto_respuesta_)

    inicio = time.time()
    resultados = evaluar_respuesta(query, texto_respuesta)
    duracion = round(time.time() - inicio, 2)

    with st.expander("📊 Evaluación automática"):
        if resultados:
            col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
            col1.metric("BLEU", resultados["BLEU"])
            col2.metric("ROUGE-L", resultados["ROUGE-L"])
            col3.metric("METEOR", resultados["METEOR"])
            col4.markdown(f"⏱️ Tiempo estimado: **{duracion} s**")
            st.markdown("---")
            st.markdown("**Relevancia estimada:** basada en la cobertura semántica y estructura esperada.")
            st.markdown("**Referencia esperada:**")
            st.info(resultados["respuesta_referencia"])
        else:
            st.warning("No se encontró una respuesta de referencia para esta pregunta en el archivo CSV.")
    
    # --- Historial acumulativo de métricas ---
    if "tabla_metricas" not in st.session_state:
        st.session_state["tabla_metricas"] = []

    if mostrar_metricas and resultados:
        # Agrega solo si es una pregunta nueva o cambia el texto de la respuesta
        fila = {
            "Pregunta": query,
            "BLEU": resultados["BLEU"],
            "ROUGE-L": resultados["ROUGE-L"],
            "METEOR": resultados["METEOR"],
            "Tiempo (s)": duracion,
            "Referencia": resultados["respuesta_referencia"],
            "Respuesta generada": texto_respuesta
        }
        # No duplicar si ya existe pregunta idéntica + respuesta igual
        ya_registrada = any(
            (f["Pregunta"] == query and f["Respuesta generada"] == texto_respuesta)
            for f in st.session_state["tabla_metricas"]
        )
        if not ya_registrada:
            st.session_state["tabla_metricas"].append(fila)

        # Mostrar la tabla acumulada si hay métricas
        if st.session_state["tabla_metricas"]:
            st.markdown("## 📊 Historial de métricas de las preguntas")
            df_metricas = pd.DataFrame(st.session_state["tabla_metricas"])
            df_metricas = pd.DataFrame(st.session_state["tabla_metricas"])[::-1]  # Invierte el orden
            st.dataframe(df_metricas, use_container_width=True)

