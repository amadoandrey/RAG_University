import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv(override=True)


DOC_DIR = os.getenv("DOC_DIR")
PERSIST_DIR = os.getenv("PERSIST_DIR")

print(f"DOC_DIR IMPRESO: {DOC_DIR}")
print(f"PERSIST_DIR IMPRESO: {PERSIST_DIR}")