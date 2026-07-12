from langchain_openai import ChatOpenAI
from src.ragState import RAGState
from src.vector_store import retrieve_docs


def respond_or_tool_call(state: RAGState):
    llm = ChatOpenAI()
    llm_with_tools = llm.bind_tools([retrieve_docs])
    response = llm_with_tools.invoke(input = state['messages'])
    answer = response.content
    return RAGState(
        messages= [response],
        documents= None,
        answer = str(answer)
    )