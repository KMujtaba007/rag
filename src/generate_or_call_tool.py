from langchain_groq import ChatGroq
from src.ragState import RAGState
from src.vector_store import retrieve_docs
from src.config_loader import load_config

config = load_config(path = 'config/llm_config.yaml').load()
model = config['model']
temperature = config['temperature']


def respond_or_tool_call(state: RAGState):
    llm = ChatGroq(model= model, 
                   temperature= temperature)
    llm_with_tools = llm.bind_tools([retrieve_docs])
    response = llm_with_tools.invoke(input = state['messages'])
    answer = response.content
    return RAGState(
        messages= [response],
        documents= None,
        answer = str(answer)
    )