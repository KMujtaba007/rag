from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    def __init__(self):
        #self.url = 'http://34.207.216.209:11434'
        self.model = 'openai/gpt-oss-120b'
        self.temperature = 0.2
    
    def get_llm(self):
        llm = ChatGroq(model= self.model,
                        temperature = self.temperature
                        )
        return llm