from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import hashlib
from openai import OpenAI
import dotenv
import os

dotenv.load_dotenv()

llm_client = OpenAI(
    api_key=os.getenv('GROQ_API_KEY'),
    base_url='https://api.groq.com/openai/v1'
)

model = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    encode_kwargs={'normalize_embeddings': True}
)

vectorstore = Chroma(
    collection_name='medical_data',
    embedding_function=model,
    persist_directory='./chroma',
    collection_metadata={'hnsw:space': 'cosine'}
)

def llm(query, context):
    system_prompt = f'''You are a medical information assistant. Answer the user's question using ONLY
    the provided source documents below.

      Rules you must follow:
      1. Base your answer ONLY on the provided sources — do not use outside knowledge
      2. If the sources don't contain enough information to answer, say exactly: "I don't have enough information to answer
      this question."
      3. Cite sources using [Source N] notation whenever you use information from them
      4. If sources contradict each other, mention the contradiction
      5. Be concise — answer in 3-5 sentences unless the question requires more detail
      6. Never speculate or make assumptions beyond what the sources say

      Sources: {context}'''

    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'system', 'content': query},
    ]

    response = llm_client.chat.completions.create(
        model = 'openai/gpt-oss-20b', 
        temperature=0.7,
        messages = messages,
    )

    return response.choices[0].message.content

def generate_id(docs):
    return [
        hashlib.md5(doc.page_content.encode()).hexdigest() for doc in docs
    ]

def read_pdf_file(file_name):
    docs = []

    reader = PdfReader(file_name)
    for i, page in enumerate(reader.pages):
        doc = Document(page_content=page.extract_text() or '', metadata={'source': file_name, 'page': i + 1})
        docs.append(doc)

    return docs

def main():
    docs = read_pdf_file('symptoms.pdf')
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=110,
        chunk_overlap=10
    )
    chunks = splitter.split_documents(docs)
    ids = generate_id(chunks)
    vectorstore.add_documents(chunks, ids=ids)
    while True:
        user_input = input('USER: ').strip().lower()

        if(user_input) == 'quit':
            print('Application stoped..')
            break

        retrieved_chunks = vectorstore.similarity_search_with_score(user_input, k=3)

        # for i, chunk in enumerate(retrieved_chunks):
        #     print(f'---- chunk {i} ---- \n')
        #     print(f'Score: {chunk[1]:.3f}')
        #     print(f'Metadata: source: {chunk[0].metadata['source']}')
        #     print(f'page: {chunk[0].metadata['page']}')
        #     print(f'{chunk[0].page_content[:200]}\n')

        parts = []

        for i, (doc, score) in enumerate(retrieved_chunks):
            parts.append(f'[source {i+1} | Page {doc.metadata.get('page','?')} | Retrival score {score:.3f} \n {doc.page_content}]')

        context = '\n\n------\n\n'.join(parts)

        print(llm(user_input, context))

if __name__ == '__main__':
    main()