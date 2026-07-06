from langchain_core.documents import Document
from typing import List
from pydantic import BaseModel

class RAGState(BaseModel):
    '''State object for RAG Workflow'''
    question: str
    retrieved_docs: List[Document] = []
    answer: str = ''