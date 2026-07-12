import streamlit as st 
from src.builder import run_graph
from src.ragState import RAGState

st.title('Enterprise Knowledge Assistant')

with st.form('search_form'):
    question = st.text_input('Enter your question',
    placeholder='What do you want to know?')
    submit_button = st.form_submit_button("Search")
    if submit_button and question:
        with st.spinner('Searching...'):
            output = run_graph(query= question)
        st.markdown('Answer:')
        st.success(output['answer'])
        if output['documents'] is not None:
            st.markdown('Sources:')
            sources = [doc.metadata['source'] for doc in output['documents']]
            sources = '\n\n'.join(sources)
            st.success(sources)
