from src.data_ingestion.document_processor import DocumentProcessor
from src.config.config import load_config
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

config = load_config(path = 'embedding_config.yaml')
config = config.load()['environment']
model = config['model']
search_type = config['search_type']
search_result = config['search_result_count']
class VectorStore:
    '''Create Vector Store and Retriever'''
    def __init__(self, model: str = model):
        self.embedding = HuggingFaceEmbeddings(model = model)
        self.retriever = None
    
    def __create_retriever(self):
        '''Create retriever'''
        processor = DocumentProcessor()
        documents = processor.load_documents()
        vector_store = Chroma.from_documents(embedding_function= self.embedding,
                                             documents = documents)
        retriever = vector_store.as_retriever(search_type = search_type,
                                  search_kwargs={
                                      'k': search_result
                                  })
        return retriever
    
    def get_retriever(self):
        '''Get retriever'''
        if self.retriever is None:
            self.retriever = self.__create_retriever()
        return self.retriever