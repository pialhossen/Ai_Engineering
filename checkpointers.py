import dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from langgraph.checkpoint.postgres import PostgresSaver
from langchain.agents.middleware import SummarizationMiddleware

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

search_tool = TavilySearch(max_results=1)

class Agent:
    def __init__(self, checkpointer: PostgresSaver):
        self.llm = ChatGroq(
            model='openai/gpt-oss-20b',
            temperature=0.5,
        )

        self.agent = create_agent(
            model=self.llm,
            tools=[get_weather, search_tool],
            system_prompt='You are a helpful assistant. give short answers.',
            checkpointer=checkpointer,
            middleware=[
                SummarizationMiddleware(
                    model=self.llm,
                    trigger=("tokens", 500),
                    keep=("messages", 5),
                )
            ],
        )

    def chat_loop(self):
        active_session = "session_1"
        while True:
            user_input = input(f'({active_session})USER: ').strip()

            if user_input.lower() == 'quit':
                break
            if user_input.lower() == 'ch':
                active_session = "session_2" if active_session == "session_1" else "session_1"
                print('Conversion Changed')
                continue

            config = {
                'configurable': {
                    'thread_id': active_session
                }
            }
            response = self.agent.invoke({'messages': [HumanMessage(user_input)]}, config=config)

            print(f'({active_session})Bot: {response['messages'][-1].content}')

        config_history = {
            'configurable': {
                'thread_id': active_session
            }
        }
        state = self.agent.get_state(config_history)
        price(state.values['messages'])

def main():
    db_url = os.getenv('DATABASE_URL')
    with PostgresSaver.from_conn_string(db_url) as checkpointer:
        checkpointer.setup()
        agent = Agent(checkpointer=checkpointer)
        agent.chat_loop()

if __name__ == '__main__':
    main()