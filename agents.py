import dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.agents import create_agent

dotenv.load_dotenv()

llm = ChatGroq(
    model='openai/gpt-oss-20b',
    temperature=0.5,
)

def get_weather(city: str) -> dict:
    '''get the current weather for a city
    Arg:
        city: the name of the city
    '''
    print('---TOOL CALL---')
    return {'city': city, 'weather': 'sunny, 26'}

class Agent:
    def __init__(self):
        self.llm = ChatGroq(
            model='openai/gpt-oss-20b',
            temperature=0.5,
        )
        self.messages = [
            SystemMessage('Give short answers, use the provided tools whenever needed')
        ]

        self.agent = create_agent(
            model=self.llm,
            tools=[get_weather]
        )

    def chat_loop(self):
        while True:
            user_input = input('USER: ').strip().lower()

            if(user_input) == 'quit':
                break
            self.messages.append(HumanMessage(user_input))
            response = self.agent.invoke({'messages': self.messages})

            self.messages.append(response['messages'][-1])

            print(f'Bot: {response['messages'][-1].content}')

def main():
    agent = Agent()
    agent.chat_loop()

if __name__ == '__main__':
    main()