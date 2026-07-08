from langchain_core.documents import Document
from typing import List
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from src.config.config import load_config
from langchain_community.document_loaders import (DirectoryLoader,
                                                  TextLoader, 
                                                  PyPDFDirectoryLoader, 
                                                  UnstructuredWordDocumentLoader)

# Load the retriever config
retrieval_config = load_config(path = 'retrieval_config.yaml')
retrieval_config = retrieval_config.load()['processor']

# Store chunk size and overlap
chunk_size = retrieval_config['chunk_size']
chunk_overlap = retrieval_config['chunk_overlap']
multi_threading = retrieval_config['multi-threading']
default_path = retrieval_config['default_path']

class DocumentProcessor:
    '''Ingests the documents and splits them into chunks'''
    def __init__(self, 
                 chunk_size: int = chunk_size, 
                 chunk_overlap: int = chunk_overlap,
                 multithreading: bool = multi_threading,
                 path: str = default_path):
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.multi_threading = multi_threading
        self.path = path
    
    def __load_text_files(self) -> List[Document]:
        dir_loader = DirectoryLoader(path = self.path,
                        glob = '**/[!.]*.txt',
                        loader_cls = TextLoader,
                        use_multithreading= self.multi_threading)
        documents = dir_loader.load()
        return documents
    
    def __load_pdf_documents(self) -> List[Document]:
        dir_loader = PyPDFDirectoryLoader(path = self.path,
                             glob = "**/[!.]*.pdf")
        documents = dir_loader.load()
        return documents
    
    def __load_word_documents(self) -> List[Document]:
        path = Path(self.path)
        word_files = [file for file in path.glob('*.docx')]
        list_of_documents = [UnstructuredWordDocumentLoader(file_path = word_file).load() for word_file in word_files]
        documents: List[Document] = []
        for document in list_of_documents:
            documents.extend(document)
        return documents
    
    
    def load_documents(self) -> List[Document]:
        '''Loads all the documents'''
        documents:List[Document] = []
        documents.extend(self.__load_text_files())
        documents.extend(self.__load_pdf_documents())
        documents.extend(self.__load_word_documents())
        return documents