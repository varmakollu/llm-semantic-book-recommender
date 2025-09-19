# Semantic Book Recommender Using Large Language Models (LLMs)

## Overview
This project builds an intelligent book recommendation system powered by large language models (LLMs) and vector search. Unlike traditional keyword-based recommenders, this system understands the semantic meaning of user queries and book descriptions to deliver highly relevant book suggestions.

Users can search for books using natural language queries like "a book about a person seeking revenge" and apply filters by category (e.g., fiction, non-fiction) and sentiment/tone (e.g., suspenseful, joyful). The system leverages the power of semantic embeddings, zero-shot classification, and sentiment analysis to provide richer recommendations.

***

## Features

- **Text Data Cleaning:** Preprocesses book metadata (title, description, author) to ensure quality input for downstream tasks.
- **Semantic Search:** Converts book descriptions and user queries into vector embeddings for efficient similarity search using a vector database.
- **Zero-shot Classification:** Classifies books into categories like fiction or non-fiction without needing labeled training data, enabling dynamic filtering.
- **Sentiment Analysis:** Extracts emotional tone from book texts (e.g., suspense, joy, sadness) to enable tone-based sorting.
- **Interactive Web App:** A user-friendly interface built with Gradio allowing real-time book recommendations from natural language queries and filters.
- **LLM Providers:** Utilizes models from Hugginghub and Perplexity for embeddings, classification, and sentiment analysis.

***

## Project Structure

| Component                     | Description                                | File / Notebook               |
|-------------------------------|--------------------------------------------|------------------------------|
| Data Cleaning                 | Cleans and preprocesses raw book data      | `data-exploration.ipynb`      |
| Semantic Search & Vector DB  | Creates vector embeddings and performs search | `vector-search.ipynb`         |
| Zero-shot Text Classification | Classifies books into categories            | `text-classification.ipynb`   |
| Sentiment & Emotion Analysis | Analyzes emotional tone of books            | `sentiment-analysis.ipynb`    |
| Web Application Interface    | Gradio app for interactive recommendation  | `gradio-dashboard.py`         |

***

## Getting Started

### Prerequisites

- Python 3.7+
- Required Python libraries:
  ```bash
  pip install pandas numpy torch sentence-transformers transformers gradio
  ```

### Running the Web App

1. Prepare your book dataset with metadata (title, author, description, categories).
2. Run the Gradio dashboard:
   ```bash
   python gradio-dashboard.py
   ```
3. Access the web interface in your browser to enter natural language queries and see recommendations.

***

## How It Works

1. **Preprocessing:** Raw book data is cleaned and structured.
2. **Embedding:** Text data is transformed into vector embeddings using pretrained LLM embedding models.
3. **Vector Search:** User queries are also embedded, and the system finds the most semantically similar books using vector similarity search.
4. **Classification:** LLM zero-shot classification categorizes books dynamically without the need for labeled data.
5. **Sentiment Analysis:** Emotional tone is extracted from text allowing filtering and sorting by mood.
6. **Frontend:** Gradio provides an easy-to-use interface to interact with the backend models seamlessly.

***

## Notes

- You may need API keys or access tokens for Hugginghub or Perplexity LLM services depending on the models selected.
- The system can be extended to include more fine-grained categories, user preferences, and additional metadata.
- Embeddings and vector database indexing may take time for large datasets but allow fast recommendation queries afterward.

***

## References

This project is inspired by and built upon recent tutorials and example repos showing how to combine semantic search, zero-shot classification, sentiment analysis, and Gradio apps for book recommendation systems

- https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata
- https://github.com/afraAntara/Semantic_Book_Recommender
- https://www.freecodecamp.org/news/build-a-semantic-book-recommender-using-an-llm-and-python
- https://github.com/t-redactyl/llm-semantic-book-recommender
