from langgraph.graph import StateGraph, END
from src.state.rag_state import RAGState
from src.nodes.nodes import RAGNodes


class GraphBuilder:
    '''Build and manages LangGraph workflow'''
    def __init__(self, retriever, llm):
        '''
        Initialized graph builder
        Args:
            retriever: Instance of Document Retriever
            llm: Large Languague Model'''
        self.nodes = RAGNodes(llm = llm, retriever = retriever)
        self.graph = None
        
    def build(self):
        # Create graph
        builder = StateGraph(state_schema = RAGState)
        
        # Add nodes
        builder.add_node('retriever',self.nodes.retrieve_document)
        builder.add_node('responder',self.nodes.generate_answer)
        
        # Add Start point
        builder.set_entry_point('retriever')
        
        # Add Edges
        builder.add_edge(start_key = 'retriever', end_key = 'responder')
        
        # Add finish point
        builder.add_edge(start_key= 'responder', end_key = END)
        
        # Compile graph
        self.graph = builder.compile()
        
        return self.graph
    
    def run(self, question: str) -> dict:
        if self.graph is None:
            self.graph = self.build()
        initial_state = RAGState(question= question)
        return self.graph.invoke(initial_state)