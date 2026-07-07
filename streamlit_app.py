import streamlit as st
from Config import Config
from src.graph_builder.graph_builder import GraphBuilder
from src.vector_store.vectorstore import VectorStore
from src.document_ingestion.document_processor import DocumentProcessor
from langchain_ollama import ChatOllama
import time

@st.cache_resource
def initialize_rag():
        processor = DocumentProcessor()
        loaded_documents = processor.load_all_documents()
        st.write(f"Successfully loaded {len(loaded_documents)} documents chunks")

        vector_store = VectorStore()
        vector_store.create_retriever(documents = loaded_documents)
        retriever = vector_store.get_retriever()
        llm = Config()
        llm = llm.get_llm()

        # llm = ChatOllama( base_url= 'http://34.207.216.209:11434',
        #         model = 'llama3.2',
        #         temperature = 0.2)

        builder = GraphBuilder(retriever = retriever, llm = llm)
        builder.build()
        return builder

def main():
        with st.spinner(text = 'Initializing the RAG'):
                builder = initialize_rag()
        st.success('RAG is ready. You can now search...')
        with st.form('search_form'):
                question = st.text_input('Enter your question',
                              placeholder='What do you want to know?')
                submit_button = st.form_submit_button("Search")
        if submit_button and question:
                # start_time = time.time()
                with st.spinner('Searching...'):
                        output = builder.run(question= question)
                # end_time = time.time()
                st.markdown('Answer:')
                st.success(output['answer'])
                # time_taken = end_time - start_time
                # st.write(f'Result in: {time_taken:. 3f} seconds')
main()