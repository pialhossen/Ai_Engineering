from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEMbeddings


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
    print(len(chunks))
    print(chunks[0])

if __name__ == '__main__':
    main()