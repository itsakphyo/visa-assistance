import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import CharacterTextSplitter

from langchain_community.document_loaders import TextLoader

# Initialize the ChatOllama model with "llama3.1" or can use ChatGPT with API Key
model = ChatOllama(model="llama3.1")

# Initialize the HuggingFaceEmbeddings transformer with a specific model for sentence embeddings
llm_transformer = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")

# Define paths for the book and database files
current_path = os.path.dirname(os.path.abspath(__file__))
book_path = os.path.join(current_path, "ThaiVisa.txt")
db_path = os.path.join(current_path, "db", "bkk_visa_db")


# Check if the database path exists; if not, create and populate it
if not os.path.exists(db_path):
    if not os.path.exists(book_path):
        raise FileNotFoundError(
            f"The file {book_path} is not exist. Please check the folder path."
        )
    
    loader = TextLoader(book_path, encoding="utf8")
    documents = loader.load()

    text_spliter = CharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    splitted_text = text_spliter.split_documents(documents)

    # Create a vector store from the document chunks and persist it to the database path
    db = Chroma.from_documents(
        splitted_text, llm_transformer, persist_directory=db_path)
    print("\n--- Finished creating vector store ---")

else:
    # If the database already exists, load it
    print("Vector store already exists. No need to initialize.")
    db = Chroma(
            persist_directory=db_path,
            embedding_function=llm_transformer,
        )
    

# Set up the retriever to search for similar documents
retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 7, "score_threshold": 0.1},
        )

# Define the system prompt for contextualizing user questions
contextualize_q_system_prompt = (
    "Analyze the latest user question, ensuring it can be understood without additional context from the chat history. Reformulate the question to be clear and standalone. Provide only the rephrased question without answering it."
)

# Create a prompt template for contextualizing user questions
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a history-aware retriever using the contextualizing prompt
history_aware_retriever = create_history_aware_retriever(
    model, retriever, contextualize_q_prompt
)

# Define the system prompt for answering questions
qa_system_prompt = (
    "You are an assistant tasked with answering the question using the provided context if necessary. If the answer is unknown or not covered by the context, state that you do not know. Keep your response clear and concise, with a maximum of three sentences. Here is the context: {context}"
)

# Create a prompt template for answering questions
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a question-answering chain using the defined prompt
question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

# Create a retrieval chain that combines the history-aware retriever and the question answering chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Define the function for continual chat with the AI
def continual_chat():
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    chat_history = []  # Collect chat history here (a sequence of messages)
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        # Process the user's query through the retrieval chain
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        # Display the AI's response
        print(f"AI: {result['answer']}")
        # Update the chat history
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result["answer"]))


# Main function to start the continual chat
if __name__ == "__main__":
    continual_chat()