import os
from functools import lru_cache
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

@lru_cache(maxsize=1)
def get_embeddings():
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if token:
        try:
            return HuggingFaceEndpointEmbeddings(
                model="sentence-transformers/all-MiniLM-L6-v2",
                huggingfacehub_api_token=token,
            )
        except Exception:
            pass
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def build_vector_store(transcript: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = splitter.split_text(transcript)
    if not chunks:
        chunks = [transcript]

    docs = [
        Document(page_content=chunk, metadata={'chunk_index': i})
        for i, chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name='video_transcript',
    )
    return vector_store

def load_vector_store():
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name='video_transcript',
        embedding_function=embeddings,
    )
    return vector_store

def get_retriever(vector_store, k: int = 3):
    return vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={"k": k}
    )