from langgraph.graph import StateGraph, START, END
from src.ragState import RAGState
from src.generate_or_call_tool import respond_or_tool_call
from src.vector_store import get_documents
from src.response import response
from langchain.messages import HumanMessage
from IPython.display import display, Image

def build_graph():
    builder = StateGraph(state_schema= RAGState)
    builder.add_node(respond_or_tool_call)
    builder.add_node(get_documents)
    builder.add_node(response)

    builder.add_edge(start_key = START, end_key = 'respond_or_tool_call')
    
    def route_on_tool_call(state: RAGState):
        last_message = state['messages'][-1]
        if getattr(last_message, 'tool_calls', None):
            return 'tool'
        return END
    
    builder.add_conditional_edges(
        source = 'respond_or_tool_call',
        path = route_on_tool_call,
        path_map={
            'tool':'get_documents',
            END: END
        }
    )

    builder.add_edge('get_documents', 'response')
    builder.add_edge('response', END)
    return builder.compile()

def show_graph(graph):
    return display(Image(graph.get_graph().draw_mermaid_png()))

def run_graph(graph, query: str):
    init_state = RAGState(
        messages= [HumanMessage(content = query)],
        documents = None,
        answer = None
    )
    return graph.invoke(input = init_state)