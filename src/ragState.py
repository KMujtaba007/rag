from langgraph.graph import MessagesState
from typing_extensions import List
from langchain_core.documents import Document

class RAGState(MessagesState):
    documents: List[Document] | None
    answer: str | None