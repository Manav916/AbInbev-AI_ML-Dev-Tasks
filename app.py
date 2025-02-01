from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple, Optional
import uvicorn
from project import run_chatbot, create_vector_store, load_docs_with_metadata
from langchain_core.documents import Document
import os
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.getcwd(), "demo_bot_data")

# Initialize FastAPI app
app = FastAPI(
    title="Chatbot API",
    description="API for interacting with the document-based chatbot",
    version="1.0.0"
)

# Define request/response models
class ChatMessage(BaseModel):
    message: str
    chat_history: Optional[List[Tuple[str, str]]] = None

class ChatResponse(BaseModel):
    response: str
    chat_history: List[Tuple[str, str]]
    sources: Optional[List[str]] = None

# Initialize vector store on startup
@app.on_event("startup")
async def startup_event():
    global vector_store
    documents, metadatas = load_docs_with_metadata(DATA_DIR)
    chunks = [Document(page_content=doc, metadata=meta) for doc, meta in zip(documents, metadatas)]
    vector_store = create_vector_store(chunks)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_message: ChatMessage):
    try:
        # Run the chatbot
        chat_history, sources = run_chatbot(
            message=chat_message.message,
            chat_history=chat_message.chat_history if chat_message.chat_history else []
        )
        
        # Get the latest response
        latest_response = chat_history[-1][1]
        
        # Extract sources if available
        # sources = []
        # if hasattr(latest_response, 'metadata') and 'source' in latest_response.metadata:
        #     sources = [latest_response.metadata['source']]
        
        return ChatResponse(
            response=latest_response,
            chat_history=chat_history,
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the Chatbot API"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 