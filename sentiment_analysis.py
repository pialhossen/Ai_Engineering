import dotenv
import os
from openai import OpenAI

dotenv.load_dotenv()

class Analyzer:
    def __init__(self):
        self.provider_client = OpenAI(
            api_key=os.getenv('GROQ_API_KEY'),
            base_url='https://api.groq.com/openai/v1'
        )

        self.messages = [{
            'role': 'system',
            'content': 'Classify the sentiment of the text as positive, negative, or neutral'
        }]

    def sentiment_analysis(self, text):
        response = self.provider_client.chat.completions.create(
            model = 'openai/gpt-oss-20b', 
            temperature=0.1,
            messages = [
                *self.messages,
                {
                    'role': 'user', 'content': text
                }
            ],
        )
        return response.choices[0].message.content

def main():
    analyzer = Analyzer()
    ai_response = analyzer.sentiment_analysis('this movie is very bad')
    print(ai_response)

if __name__ == '__main__':
    main()