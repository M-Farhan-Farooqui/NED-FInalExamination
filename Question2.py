import os

from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import json
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import time
from langchain.embeddings import OpenAIEmbeddings

load_dotenv()

# Load data from JSON file
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Create embeddings and store in a vector database
def create_embeddings_and_store(data):
    product_texts = []
    for p in data:
        attribute_text = ' '.join([
            f"{key} {value}"
            for attr in p['attributes']
            for key, value in attr.items()
        ])
        text = f"{p['title']} {p['brand']} {p['description']} {attribute_text}"
        product_texts.append(Document(page_content=text))

    
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
    db = FAISS.from_documents(product_texts, embeddings)
    return db

# Set up the RAG pipeline
def setup_rag_pipeline(db):
    llm = OpenAI(openai_api_key=os.environ['OPENAI_API_KEY'], temperature=0.9)  
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                            chain_type="stuff",
                                            retriever=db.as_retriever())
    return qa_chain

# Run the RAG pipeline
def run_rag_pipeline(qa_chain, query):
    result = qa_chain.run(query)
    return result

# Main function
def main():
    # Load data
    data = load_data("catalog.json")
    max_catalog_size = int(os.environ.get("MAX_CATALOG_SIZE", 100))
    data = data[:max_catalog_size]

    # Create embeddings and store in vector database
    db = create_embeddings_and_store(data)

    # Set up RAG pipeline
    qa_chain = setup_rag_pipeline(db)

    # Run RAG pipeline
    queries = [
        "What is the main topic of this document?",
        "What brands are available?",
        "Can you list some products with cotton fabric?",
    ]

    for query in queries:
        result = run_rag_pipeline(qa_chain, query)
        print(f"Query: {query}")
        print(f"Result: {result}")
        print("-" * 20)

    print(result)

if __name__ == "__main__":
    main()
    