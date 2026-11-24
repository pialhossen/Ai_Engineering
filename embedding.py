import dotenv
from sentence_transformers import SentenceTransformer, util

dotenv.load_dotenv()

SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def find_closest(query, sentences):
    embeddings = model.encode(sentences)
    query_embedding = model.encode(query)

    closest_index = 0
    closest_score = -1

    for index, vector in enumerate(embeddings):
        similarity = util.cos_sim(vector, query_embedding).item()

        if similarity > closest_score:
            closest_index = index
            closest_score = similarity
    return sentences[closest_index]

def main():
    sentences = [
        "That is a happy person",
        "That is a happy dog",
        "That is a very happy person",
        "Today is a sunny day"
    ]

    query = "I am feeling great today"

    embeddings = model.encode(sentences)
    query_embedding = model.encode(query)

    result = util.semantic_search(query_embedding, embeddings, top_k=2)
    print(result)

if __name__ == '__main__':
    main()