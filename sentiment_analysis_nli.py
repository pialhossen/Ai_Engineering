import dotenv
import os
from transformers import pipeline
from openai import OpenAI

dotenv.load_dotenv()

class Analyzer:
    def __init__(self):
        self.classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")

    def sentiment_analysis(self, text):
        labels = ['finance', 'health','sport']
        result = self.classifier(text, candidate_labels=labels)
        return result

def main():
    analyzer = Analyzer()
    sentiment = analyzer.sentiment_analysis('I had to take the economic class.')
    print(sentiment)

if __name__ == '__main__':
    main()