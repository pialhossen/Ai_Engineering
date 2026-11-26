import os
import dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

dotenv.load_dotenv()

class User(BaseModel):
    name: str = Field(description='The name of the user')
    age: int = Field(description='The age of the user')
    companies: list[str] = Field(description='The companies the user')
    net_worth: str = Field(description='The net worth of the user')

class Agent:
    def __init__(self):
        self.llm = ChatGroq(
            model='openai/gpt-oss-20b',
            temperature=0.1,
        )
    def chat(self):
        # PromptTemplate example

        # template = '''Dear {user_name}
        # Apologizing for the bad experience, we will set an appointment to further discuss your problem
        
        # Sincerely
        # {brand_name}
        # '''
        # prompt = PromptTemplate(input_variables=["user_name", "brand_name"], template=template)

        # prompt.format(user_name='Pial', brand_name='Google')

        # response = self.llm.invoke("hello")
        # print(response.content)

        # ChatPromptTemplate example
        # history = [
        #     HumanMessage('I like you'),
        #     AIMessage('>>>আমি তোমাকে পছন্দ করি<<<'),
        #     HumanMessage('I hate you'),
        #     AIMessage('>>>আমি তোমাকে ঘৃণা করি<<<'),
        # ]

        # prompt = ChatPromptTemplate.from_messages([
        #     ('system', 'you are a translator from {origin} to {to}'),
        #     MessagesPlaceholder(variable_name='conversation_history'),
        #     ('human','{input}')
        # ])
        # chain = prompt | self.llm

        # response = chain.invoke({
        #     'origin': 'English',
        #     'conversation_history': history,
        #     'to': 'Bangla', 
        #     'input': 'I love you'
        # })
        # print(response.content)

        structured_llm = self.llm.with_structured_output(User, method='json_schema')
        response = structured_llm.invoke("Elon Reeve Musk (/ˈiːlɒn/ ⓘ; born June 28, 1971) is an industrialist. He is the chief executive officer and largest shareholder of Tesla and SpaceX and has been the wealthiest person in the world since 2025, becoming the only trillionaire in history (in terms of US dollars) in 2026; as of November 2026, Forbes estimates his net worth to be $849.7 billion.")
        print(response)
        

def main():
    agent = Agent()
    agent.chat()

if __name__ == '__main__':
    main()
