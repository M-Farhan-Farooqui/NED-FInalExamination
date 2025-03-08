
## **QUESTION 2** 
# RAG Pipeline for Product Catalog

This project implements a Retrieval-Augmented Generation (RAG) pipeline to answer questions about a product catalog. It uses Langchain to orchestrate the pipeline, OpenAI's models for both embeddings and language generation, and FAISS for efficient vector storage and retrieval.

## **Approach** 

The core idea is to create a system that can answer questions about products based on the information available in a structured catalog. The RAG pipeline consists of the following steps:

1.  **Data Loading:** The product catalog is loaded from a JSON file (`catalog.json`).
2.  **Embedding Generation:**  Each product's title, brand, description, and attributes are combined into a single text string.  OpenAI's `text-embedding-ada-002` model (accessed via `OpenAIEmbeddings` in Langchain) is used to generate embeddings for these text strings. These embeddings capture the semantic meaning of the product information.
3.  **Vector Storage:** The generated embeddings are stored in a FAISS (Facebook AI Similarity Search) index. FAISS allows for efficient similarity search, which is crucial for retrieving relevant products based on a user's query.
4.  **Retrieval:** When a user asks a question, the question is also embedded using the same OpenAI embedding model.  The FAISS index is then queried to find the most similar product embeddings. The corresponding product information is retrieved.
5.  **Generation:** The retrieved product information is combined with the original question and fed into OpenAI's `gpt-3.5-turbo` model (accessed via `OpenAI` in Langchain). The model generates an answer based on the provided context.

## Models Used

*   **Embeddings:** OpenAI's `text-embedding-ada-002` model (via `OpenAIEmbeddings`). This model is used to generate embeddings for both the product information and the user's queries.
*   **Language Model:** OpenAI's `gpt-3.5-turbo` model (via `OpenAI`). This model is used to generate the final answer to the user's query, conditioned on the retrieved product information.
*   **Vector Database:** FAISS (Facebook AI Similarity Search). This is used for efficient storage and retrieval of the product embeddings.

## Libraries Used

*   Langchain: For orchestrating the RAG pipeline.
*   FAISS: For vector storage and similarity search.
*   OpenAI: For embeddings and language generation.
*   dotenv: For loading environment variables from a `.env` file.
*   pydantic: For data validation and settings management.

## Environment Variables

The following environment variables need to be set:

*   `OPENAI_API_KEY`: Your OpenAI API key.
*   `MAX_CATALOG_SIZE` (optional): The maximum number of products to load from the catalog. Defaults to 100.

## Usage

1.  Clone the repository.
2.  Install the required packages: `pip install -r requirements.txt`
3.  Create a `.env` file in the root directory and set the environment variables.
4.  Run the `Q2.py` script: `python Q2.py`

The script will load the product catalog, create the embeddings, set up the RAG pipeline, and then run a few example queries.  The results will be printed to the console.
