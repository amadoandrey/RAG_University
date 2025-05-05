import os
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv(override=True)

DOC_DIR = os.getenv("DOC_DIR")
PERSIST_DIR = os.getenv("PERSIST_DIR")

print(f"DOC_DIR: {DOC_DIR}")
print(f"PERSIST_DIR: {PERSIST_DIR}")



def load_pdfs(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(directory, filename)
            loader = PyPDFLoader(filepath)
            pages = loader.load()  # Load the pages
            # Create a Document object with the combined content of all pages
            text = " ".join([page.page_content for page in pages])
            metadata = {"source": filename}
            document = Document(page_content=text, metadata=metadata)
            documents.append(document)
    return documents

documents = load_pdfs(DOC_DIR)

# ✅ Imprimir nombres de los documentos
print("📂 **Documentos cargados:**")
for doc in documents:
    print(f"- {doc.metadata['source']}")
