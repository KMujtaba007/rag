from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (DirectoryLoader, 
                                                  TextLoader, 
                                                  PyPDFDirectoryLoader, 
                                                  UnstructuredWordDocumentLoader)
from typing import List
from pathlib import Path
from src.config_loader import load_config

config = load_config('config/retrieval_config.yaml').load()['processor']
chunk_size = config['chunk_size']
chunk_overlap = config['chunk_overlap']

def ingest(path: str) -> List[Document]:
    text_loader = DirectoryLoader(path = path,
                    glob = '**/[!.]*.txt',
                    loader_cls = TextLoader)
    pdf_loader = PyPDFDirectoryLoader(path = path,
                                      glob= '**/[!.]*.pdf')
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    documents = []
    source_path = Path(path)
    word_files = [file for file in source_path.glob('*.docs')]
    for file in word_files:
        loader = UnstructuredWordDocumentLoader(file_path= file)
        docs = loader.load_and_split(text_splitter= splitter)
        documents.extend(docs)
    
    text_documents = text_loader.load_and_split(text_splitter = splitter)
    documents.extend(text_documents)
    pdf_documents = pdf_loader.load_and_split(text_splitter= splitter)
    documents.extend(pdf_documents)
    return documents