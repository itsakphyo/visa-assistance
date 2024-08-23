# ChatBot with LangChain Integration

## Overview

This is a python script for a retrieval-augmented generation (RAG) conversation system with a chatbot, focused on answering questions about Thai visas.

## Features

- **Document Loading**: Load and process text documents to create a vector store.
- **Text Embeddings**: Use sentence embeddings for efficient text retrieval.
- **History-Aware Retrieval**: Analyze and rephrase user questions to improve response accuracy.
- **Question Answering**: Provide concise and contextually relevant answers based on the loaded documents.
- **Continual Chat**: Engage in ongoing conversations with the AI, with chat history management.

## Prerequisites

- Python 3.
- langchain (including langchain_ollama, langchain_huggingface, langchain_chroma, langchain_core)
- TextLoader (from langchain_community)

## Optional

- Ollama chat language model (accessible via ChatOllama class)
- lChatGPT API (can be used instead of Ollama)


## Installation

1. **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Install dependencies:**

    Create a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

    Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## How it Works:

The script first initializes a retrieval system using a factual document about Thai visas. It then utilizes a large language model (Ollama or ChatGPT) to process user queries and answer them based on the retrieved information. The system aims to provide concise and informative answers within three sentences.

## Continual Chat:

The script offers a continual chat interface where you can ask questions about Thai visas. Type your questions and the AI will respond. To exit the conversation, type "exit".

## Note:

- This is a basic example and can be further customized for different domains and functionalities.
- The accuracy of the answers depends on the quality of the factual document and the capabilities of the large language model.

## Limitations:

- The current implementation uses a single factual document. A more comprehensive knowledge base could improve the quality of responses.
- The large language model might generate incorrect or misleading information. It's essential to verify the answers from other reliable sources.