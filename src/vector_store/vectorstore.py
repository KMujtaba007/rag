from langchain_chroma import Chroma
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class VectorStore:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings()
        self.retriever = None
        self.vectorstore = None
    
    def create_retriever(self, documents: List[Document]):
        '''
        Creates retriever
        '''
        self.vectorstore = Chroma.from_documents(documents= documents,embedding=self.embeddings)
        self.retriever = self.vectorstore.as_retriever()
        
    def get_retriever(self):
        if self.retriever is None:
            raise ValueError("Please run 'create_retriever' method to create the retriever")
        return self.retriever