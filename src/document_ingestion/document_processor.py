from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (TextLoader, 
                                                  DirectoryLoader, 
                                                  PyPDFDirectoryLoader, 
                                                  UnstructuredWordDocumentLoader)

class DocumentProcessor:
    '''
    Processes the Text, Word and PDF documents
    '''
    def __init__(self, chunk_size:int = 500, chunk_overlap:int = 20):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap
        )
    
    def load_text_files(self, directory:str)-> List[Document]:
        '''
        Loads all text files in the directory
        Args:
            directory: Directory path in which all text files are present
        Returns:
            List of Document
        '''
        
        dir_loader = DirectoryLoader(path= directory,
                        glob = "**/[!.]*.txt",
                        loader_cls = TextLoader)
        documents = dir_loader.load()
        return documents
    
    def load_pdf_files(self, directory:str)-> List[Document]:
        '''
        Loads all pdf files
        Args:
            directory: Directory path in which all pdf files are present
        Returns:
            List of Document
        '''
        
        dir_loader = PyPDFDirectoryLoader(
            path = directory,
            glob = '**/[!.]*.pdf'
        )
        documents = dir_loader.load()
        return documents
    
    def load_word_files(self, directory):
        '''
        Loads all the word documents. Loads only '.doc' and '.docx' files.
        Args:
            directory: Directory path in which all word files are present.
        Returns:
            List of Document
        '''
        
        docs = []
        dir_path = Path(directory)
        word_files = [file for file in dir_path.iterdir() if file.suffix.endswith(('.doc','.docx'),)]
        for file in word_files:
            loader = UnstructuredWordDocumentLoader(file_path = file)
            documents = loader.load()
            docs.extend(documents)
        return docs
    
    def load_all_documents(self, directory:str = 'data/')->List[Document]:
        '''
        Loads all the files into a List of Document
        Args:
            directory: Path of the directory in which all files will be placed for ingestion
        Returns:
            List of Document
        '''
        dir_path = Path(directory)
        valid_files = [file for file in dir_path.iterdir() if file.suffix.endswith(('.doc','.docx','.pdf','.txt'))]
        print(f"Found {valid_files.count} valid files")
        docs = []
        docs.extend(self.load_pdf_files(directory= directory))
        docs.extend(self.load_text_files(directory = directory))
        docs.extend(self.load_word_files(directory = directory))
        return docs
    
    def split_documents(self, documents: List[Document])->List[Document]:
        '''
        Splits the documents into chunks. Uses RecursiveCharacterTextSplitter
        Args:
            documents: List of Document to split into chunks
        Returns:
            List of Document
        '''
        
        document_chunks = self.splitter.split_documents(documents= self.load_all_documents())
        return document_chunks