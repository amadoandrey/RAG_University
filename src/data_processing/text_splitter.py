from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import re

def split_texts_with_textSplitter(_documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    return text_splitter.split_documents(_documents)



def split_texts(documents):
    all_chunks = []
    for doc in documents:
        text = doc.page_content
        metadata = doc.metadata

        # Regex para extraer cada artículo
        pattern = r"(Artículo\s\d+\..*?)(?=\nArtículo\s\d+\.|\Z)"
        matches = re.findall(pattern, text, re.DOTALL)

        # Crear un nuevo documento por artículo
        for match in matches:
            chunk = Document(page_content=match.strip(), metadata=metadata)
            all_chunks.append(chunk)

    return all_chunks


def split_texts_with_mark_down(documents):
    from langchain.text_splitter import MarkdownHeaderTextSplitter
    from langchain.docstore.document import Document

    # Definimos encabezados comunes en documentos normativos
    headers_to_split_on = [
        ("#", "TÍTULO"),
        ("##", "CAPÍTULO"),
        ("###", "SECCIÓN"),
        ("####", "ARTÍCULO"),
    ]

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    # Unimos el contenido de todos los documentos
    combined_text = "\n\n".join([doc.page_content for doc in documents])

    # Aplicamos la división
    docs = splitter.split_text(combined_text)

    # Ya retorna una lista de Document con metadatos jerárquicos
    return docs
