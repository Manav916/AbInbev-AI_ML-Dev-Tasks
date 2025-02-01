# Ubuntu Chatbot

## Installation

```bash
pip install -r requirements.txt
```

## Setup environment variables

Get groq_api_key and store it in .env file

```bash
GROQ_API_KEY=your_api_key
```


## Running the project

```bash
python project.py
```

## Running the FastAPI server

```bash
uvicorn app:app --reload
```

![fastapi](https://github.com/user-attachments/assets/b340b5aa-0c41-4670-bec3-1a88a8921de5)


## Deploying using docker

```bash
docker build -t ubuntu-chatbot .
docker run -p 80:80 ubuntu-chatbot
```

![docker_deploy](https://github.com/user-attachments/assets/e1c82d16-b467-42f3-a174-2550f7512004)

## Chunking

I have just used the UnstructuredMarkdownLoader with mode="elements". It splits the markdown files into elements based on the document structure (like headers, paragraphs, lists, etc.). This is a structural chunking approach rather than a semantic one. The documentation has distinct sections—like headings, subheadings, and code blocks. These sections are usually coherent on their own which is why I used this method. I also did not use any hard capping for the chunk size. Just based on my experimentation, I did not see any issues with the chunks being too large.

### Potential improvements

- trying a more controlled chunking strategy with something like RecursiveCharacterTextSplitter or MarkdownTextSplitter to get chunks of consistent size
- adding overlap between chunks to maintain context
- implementing custom chunk size and overlap parameters
- By placing a hard cap (in tokens or words), you ensure each chunk can fit into the model without truncation.

### Retrieval improvements

- I could try and experiement with domain specific embeddings.
- I would have to incorporate the metadata and the text of the chunks to create a more accurate embedding.
- I could try and use a reranker to improve the quality of the retrieved chunks.
- If certain queries appear over and over (e.g., “How do I install package X?”), we could cache the top retrieval results to serve them instantly.
- I could try and integrate a knowledge graph to improve the retrieval.
