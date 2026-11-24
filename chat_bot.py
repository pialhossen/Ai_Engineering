import dotenv
import os 
import time
from openai import OpenAI

dotenv.load_dotenv()

class Chatbot:
    def __init__(self):
        self.llm = OpenAI(
            api_key=os.getenv('GROQ_API_KEY'),
            base_url='https://api.groq.com/openai/v1'
        )

        self.messages = [{
            'role': 'system',
            'content': '''
                You are a helpful assistant. your name is JARVIS.
                give short to the point answer. no need to over Explain.
            '''
        }]

        self.total_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0

        self.iteration = 0

    def trim_messages(self, max_number = 4):
        if len(self.messages) < max_number:
            return
        system_messages = [m for m in self.messages if m['role'] == 'system']
        other_messages = [m for m in self.messages if m['role'] != 'system']

        trimed_messages = other_messages[-max_number:]

        self.messages = [*system_messages, *trimed_messages]

    def summarize_messages(self):
        if self.iteration % 5 != 0:
            return

        system_messages = [m for m in self.messages if m['role'] == 'system']
        other_messages = [m for m in self.messages if m['role'] != 'system']

        early_messages, last_messages = (other_messages[0:5], other_messages[5:])

        response = self.llm.chat.completions.create(
            model = 'openai/gpt-oss-20b', 
            messages = [
                {
                    'role': 'system',
                    'content': 'Summarize these messages concisely, keeping key facts and decisions intact'
                }, *early_messages
            ],
        )
        summarized_message = {
            'role': 'system',
            'content': f'''This is a summary of the last five messages: {response.choices[0].message.content}'''
        }

        self.messages = [*system_messages, summarized_message, *last_messages]

    def bot(self, message):
        self.messages.append({
            'role': 'user',
            'content': message
        })

        response = self.llm.chat.completions.create(
            model = 'openai/gpt-oss-20b', 
            messages = self.messages,
            stream = True,
            stream_options = {'include_usage': True},
        )

        response_text = ''
        print('CHATBOT: ', end='', flush=True)
        for chunk in response:
            if chunk.choices:
                delta = chunk.choices[0].delta.content
                if delta:
                    print(delta, end='', flush=True)
                    response_text += delta
                    time.sleep(0.05)

                if chunk.usage:
                    self.total_tokens += chunk.usage.total_tokens
                    self.input_tokens += chunk.usage.prompt_tokens
                    self.output_tokens += chunk.usage.completion_tokens

        print()

        self.messages.append({
            'role': 'assistant',
            'content': response_text
        })

        # self.trim_messages()
        self.summarize_messages()

    def chat(self):
        while True:
            self.iteration += 1
            user_input = input('USER: ').strip().lower()

            if(user_input) == 'quit':
                print('Thank you for using chatbot')
                break
            self.bot(user_input)
            

    def report(self):
        for message in self.messages:
            print(message)
        print(f'Consumed : {self.total_tokens} tokens')
        print(f'$ {(self.input_tokens * 0.05 + self.output_tokens * 0.08) / 1_000_000}')

def main():
    chatbot = Chatbot()
    chatbot.chat()
    chatbot.report()
if __name__ == '__main__':
    main()