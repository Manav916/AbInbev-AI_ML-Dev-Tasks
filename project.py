# import libraries
import os
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document
import nltk
import os
import dotenv
from groq import Groq
import warnings

for package in ['averaged_perceptron_tagger_eng', 'punkt', 'punkt_tab', 'stopwords']:
    nltk.download(package, quiet=True)
dotenv.load_dotenv()
warnings.filterwarnings("ignore")

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DATA_DIR = os.path.join(os.getcwd(), "demo_bot_data")
VECTOR_STORE_TYPE = "faiss"  
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Load embedding model
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def load_docs_with_metadata(data_dir):
    """Recursively loads markdown (.md) files and stores metadata (folder name)."""
    docs = []
    metadata = []
    count_data = 0
    for root, _, files in os.walk(data_dir):
        # folder_name = os.path.basename(root)  # Extract folder name
        for file_name in files:
            if file_name.endswith(".md"):
                file_path = os.path.join(root, file_name)
                loader = UnstructuredMarkdownLoader(file_path, mode="elements")
                data = loader.load()
                count_data += len(data)
                for chunk in data:
                    docs.append(str(chunk.page_content))
                    metadata.append(
                        {   
                            "file_name": file_name,
                            "category": chunk.metadata["category"],
                            "source": chunk.metadata["source"],
                            # "last_modified": chunk.metadata["last_modified"],
                            # "page_number": chunk.metadata["page_number"],
                            # "languages": chunk.metadata["languages"],
                            # "filetype": chunk.metadata["filetype"],
                            # "element_id": chunk.metadata["element_id"]
                        }
                    )
    print(f"Loaded {count_data} documents from {data_dir}")
    return docs, metadata

# Load documents and metadata
documents, metadatas = load_docs_with_metadata(DATA_DIR)

# Documents are already chunked from UnstructuredMarkdownLoader
chunks = [Document(page_content=doc, metadata=meta) for doc, meta in zip(documents, metadatas)]

# Create Vector Store with documents and metadata
def create_vector_store(chunks):
    texts = [chunk.page_content for chunk in chunks]
    metadata = [chunk.metadata for chunk in chunks]  # Preserve metadata
    
    if VECTOR_STORE_TYPE == "faiss":
        db = FAISS.from_texts(texts, embeddings, metadatas=metadata)
        db.save_local("faiss_store")
    # else:
    #     db = Chroma.from_texts(texts, embeddings, metadatas=metadata)
    
    return db

vector_store = create_vector_store(chunks)
retriever = vector_store.as_retriever()

# print("Vector store created with metadata!")

def run_chatbot(message, chat_history=None):
    if chat_history is None:
        chat_history = []
    
    # Initialize Groq client
    client = Groq()
    
    # Format chat history into messages format
    messages = []
    for msg in chat_history:
        messages.append({"role": "user", "content": msg[0]})
        messages.append({"role": "assistant", "content": msg[1]})
    
    # Add current message
    messages.append({"role": "user", "content": message})

    # If there are previous messages, retrieve relevant documents
    if messages:
        print("Retrieving information...", end="")
        # Get relevant documents from vector store
        docs_with_scores = vector_store.similarity_search_with_score(message, k=20)
        # Filter docs with similarity score above threshold (0.7)
        docs = [doc for doc, score in docs_with_scores if score >= 0.7]
        # Filter documents based on similarity score threshold
        context = "\n\n".join([doc.page_content for doc in docs])
        messages.append({
            "role": "system", 
            "content": f"Use this context to help answer the question: {context}"
        })

    # Generate response using Groq
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="deepseek-r1-distill-llama-70b",  # Using Mixtral model
        temperature=0.7,
        max_tokens=1024,
        stream=True,
        reasoning_format="hidden"
    )

    # Print the chatbot response
    print("\nChatbot:")
    chatbot_response = ""
    
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="")
            chatbot_response += content
            
    # Add the new exchange to chat history
    chat_history.append((message, chatbot_response))
    
    # Print document sources if available
    if messages and len(messages) > 1:
        print("\n\nSOURCES:")
        # Use a set to track unique sources
        unique_sources = set()
        for doc in docs:
            if 'source' in doc.metadata:
                unique_sources.add(doc.metadata['source'])
        
        # Print unique sources
        for source in unique_sources:
            print(f"- {source}")

    return chat_history, unique_sources


if __name__ == "__main__":
    chat_history = []
    
    while True:
        print("\nOptions:")
        print("1. Start (new conversation)")
        print("2. Continue (current conversation)")
        print("3. Stop")
        
        choice = input("\nEnter your choice (1/2/3): ").strip().lower()
        
        if choice in ['3', 'stop']:
            print("Goodbye!")
            break
            
        elif choice in ['1', 'start']:
            chat_history = []  # Clear chat history
            print("\nStarting new conversation...")
            
        elif choice in ['2', 'continue']:
            if not chat_history:
                print("\nNo existing conversation. Starting new chat...")
            # else:
            #     print("\nContinuing previous conversation...")
        
        else:
            print("\nInvalid choice. Please try again.")
            continue
            
        # Get user input
        message = input("\nYou: ").strip()
        
        if not message:
            print("Please enter a message.")
            continue
            
        # Process the message and update chat history
        chat_history, sources = run_chatbot(message, chat_history)

