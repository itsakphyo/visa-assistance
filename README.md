# ChatBot with LangChain Integration

## Overview

This is a python script for a retrieval-augmented generation (RAG) conversation system with a chatbot, focused on answering questions about Thai visas.

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
