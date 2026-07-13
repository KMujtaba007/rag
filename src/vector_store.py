from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from functools import lru_cache
from langchain.tools import tool
from src.ingest import ingest
from src.ragState import RAGState
from src.config_loader import load_config
from dotenv import load_dotenv

load_dotenv()

config = load_config('config/embedding_config.yaml').load()['environment']
search_type = config['search_type']
k = config['search_result_count']
score_threshold = config['score_threshold']
model = config['model']

docs = ingest(path = 'data')
@lru_cache(maxsize= 2)
def load_vector_store():
    '''Creates a retriever for the Vector Store'''
    embedding = HuggingFaceEmbeddings(model_name = model)
    vector_store = Chroma.from_documents(documents = docs,embedding= embedding)
    return vector_store

vector_store = load_vector_store()

def get_retriever(vector_store = vector_store):
    retriever = vector_store.as_retriever(
        search_type = search_type,
        search_kwargs = {
            'k': k,
            'score_threshold': score_threshold
        }
    )
    return retriever

@tool
def retrieve_docs():
    '''Search and return information from vector store'''
    pass

def get_documents(state: RAGState):
    retriever = get_retriever()
    retrieved_documents = retriever.invoke(input = str(state['messages'][0].content))
    return RAGState(
        messages= state['messages'],
        documents= retrieved_documents,
        answer= None
    )