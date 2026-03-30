# summarization_agent.py
import os
from openai import OpenAI
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.callstate import CallState

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def summarization_agent(state:CallState):
    transcript = state["transcript"]
    policy_context = retrieve_context(transcript)
    prompt = f"""
You are a call center assistant.

Use company policy context when summarizing transcript.

Policy:
{policy_context}

Transcript:
{transcript}

Return JSON:
- summary
- key_issue
- resolution
- action_items
- sentiment (positive, neutral, negative)
- tags (3-5 keywords)

Return JSON only.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    state["summary"] = response.choices[0].message.content
    return state
    

def retrieve_context(query):
    """this allows the system to check company policies before generating summaries."""
    # if running for the first time, we need to create FAISS index -
    # initialize the index, add data vectors, save it to disk
    
 
    folder_path = "data/policy_docs"
    # LangChain creates these two specific files
    faiss_file = os.path.join(folder_path, "index.faiss")
    pkl_file = os.path.join(folder_path, "index.pkl")
    
    if not os.path.isfile(faiss_file) or not os.path.isfile(pkl_file):
        print("Index not found. Creating a new one...")
        # This creates a list containing ONE large Document object
        loader = TextLoader("data/policy_docs/policy_docs.txt")
        raw_documents = loader.load()
        # chunk creation
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,    # Maximum characters per chunk
            chunk_overlap=100
        )
        # This creates the 'docs' list (a list of multiple Document objects)
        docs_index = text_splitter.split_documents(raw_documents)
        # initialize and create the index
        vector_store = FAISS.from_documents(docs_index, OpenAIEmbeddings())
        vector_store.save_local(folder_path)
    print("Index found! Loading...")
    db = FAISS.load_local(
        "data/policy_docs",
        OpenAIEmbeddings(), 
        allow_dangerous_deserialization=True
    )

    docs = db.similarity_search(query, k=3)

    return [doc.page_content for doc in docs]
