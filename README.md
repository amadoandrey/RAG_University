# 🤖 Chatbot RAG - Universidad Politécnico Grancolombiano

Este repositorio contiene la implementación de un sistema de **Retrieval-Augmented Generation (RAG)** para gestionar documentos académicos y administrativos de la universidad.  
El proyecto hace parte de un trabajo de grado en la **Maestría en Analítica** y utiliza técnicas de **Procesamiento de Lenguaje Natural (NLP)** para responder preguntas de forma confiable a partir de documentos previamente cargados.

---

## 📂 Estructura del proyecto

```
Repositorio_RAG_Universidad/
│
├── notebooks/
│   └── Ejecutar_Rag.ipynb        # Notebook principal para ejecución en Colab
│
├── src/
│   ├── app_loader.py             # Streamlit: carga de documentos PDF
│   ├── app_chat.py               # Streamlit: consultas RAG + métricas
│   └── utils/                    # Funciones auxiliares reutilizables
│       ├── text_processing.py    # Fragmentación y limpieza de texto
│       ├── db_utils.py           # Manejo de embeddings y ChromaDB
│       ├── evaluation.py         # Métricas automáticas (BLEU, ROUGE, METEOR)
│       └── model_utils.py        # Carga del modelo y tokenizer
│
├── data/                         # ⚠️ No subir PDFs reales (solo estructura vacía)
│   ├── BD/                       # Bases vectoriales persistidas
│   └── reference/                # Respuestas de referencia para métricas
│
├── images/
│   └── Logo_poli.jpg             # Logo de la universidad para la interfaz
│
├── requirements.txt              # Dependencias del proyecto
├── README.md                     # Este archivo
└── .gitignore                    # Ignora datos sensibles, cachés y temporales
```

---

## ⚡ Cómo ejecutar en Google Colab

1. **Clonar el repositorio**
   ```bash
   !git clone https://github.com/TU_USUARIO/Repositorio_RAG_Universidad.git
   %cd Repositorio_RAG_Universidad
   ```

2. **Instalar dependencias**
   ```bash
   !pip install -r requirements.txt
   ```

3. **Cargar documentos (app_loader)**
   ```bash
   !streamlit run src/app_loader.py --server.port 8501 --server.address 0.0.0.0 & \
   !cloudflared tunnel --url http://localhost:8501 --no-autoupdate
   ```
   👉 Esto abre un link tipo `https://xxxx.trycloudflare.com` donde podrás subir tus PDFs y generar la base vectorial en Chroma.

4. **Consultar al chatbot (app_chat)**
   ```bash
   !streamlit run src/app_chat.py --server.port 8501 --server.address 0.0.0.0 & \
   !cloudflared tunnel --url http://localhost:8501 --no-autoupdate
   ```
   👉 Esto abre otro link donde podrás hacer preguntas y visualizar métricas automáticas.

---

## 📊 Métricas de evaluación

El sistema incluye una sección de **evaluación automática** de respuestas, comparando la salida del modelo con respuestas de referencia (archivo CSV).  
Se calculan las siguientes métricas clásicas de NLP:

- **BLEU** (Bilingual Evaluation Understudy)  
- **ROUGE-L** (Recall-Oriented Understudy for Gisting Evaluation)  
- **METEOR** (Metric for Evaluation of Translation with Explicit ORdering)  

Esto permite validar la calidad de las respuestas generadas y medir la cobertura semántica.

---

## 💡 Nota académica

Este proyecto demuestra cómo integrar un **LLM en español** (ej. *Llama 3.2 3B Instruct*) con **técnicas de RAG** para responder preguntas sobre documentos institucionales.  
La modularización del código facilita la replicación, comparación de modelos y la extensión hacia otros dominios (ej. PQRS, normatividad universitaria, material académico).

---

## ⚠️ Consideraciones

- No subas **documentos privados** al repositorio (usa `.gitignore`).  
- El sistema está optimizado para correr en **Google Colab + Cloudflare Tunnel**.  
- Si usas GPU, el modelo se cargará en `torch.float16`; en CPU se recomienda ajustar a `torch.float32`.  

---

## ✨ Créditos

Trabajo de grado de la **Maestría en Analítica** – Universidad Politécnico Grancolombiano.  
Autor: *[Tu Nombre]*  
Año: 2025
