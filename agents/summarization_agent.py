from dotenv import load_dotenv
load_dotenv()

import json
import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pydantic import BaseModel, field_validator
from typing import List
from utils.callstate import CallState
from utils.logger import log_step

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

POLICY_DOCS_PATH = "data/policy_docs"
POLICY_FILE = "data/policy_docs/policy_docs.txt"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
TOP_K_RESULTS = 3


class SummarySchema(BaseModel):
    """Enforces structured json format and avoids downstream crashes."""
    summary: str
    key_issue: str
    resolution: str
    action_items: List[str]
    sentiment: str
    tags: List[str]

    @field_validator("action_items", mode="before")
    def normalize_action_items(cls, v):
        """Ensure action_items is always a list."""
        if isinstance(v, str):
            return [v]
        return v

    @field_validator("tags", mode="before")
    def normalize_tags(cls, v):
        """Ensure tags is always a list."""
        if isinstance(v, str):
            return [v]
        return v


def summarization_agent(state:CallState) -> CallState:
    """
    Summarizes call transcript using GPT-4o with RAG-based policy context.
    Retrieves relevant company policies via FAISS vector search before
    generating structured summary.

    Args:
        state: Pipeline state containing transcript to summarize

    Returns:
        Updated state with summary dict populated containing summary,
        key_issue, resolution, action_items, sentiment and tags.
        Sets error field if summarization fails.

    Raises:
        Exception: If LLM call or response parsing fails
    """
    log_step("summarization_agent", {"status": "starting"})

    transcript = state.get("transcript", "")
    policy_context = "\n\n".join(retrieve_context(transcript))
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
- action_items: must always be a JSON array of strings.
- sentiment (positive, neutral, negative)
- tags (3-5 keywords)

Return JSON only.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        raw_output = response.choices[0].message.content

        # Validate output against SummarySchema
        summary_dict = json.loads(raw_output)
        summary = SummarySchema(**summary_dict).model_dump()
        state["summary"] = summary
        log_step("summarization_agent", {"status": "complete"})

    except Exception as e:
        log_step("summarization_agent", {"status": "failed", "error": str(e)})
        state["error"] = "bad_summary"

    if "trace" not in state or state["trace"] is None:
        state["trace"] = []
    state["trace"].append("summarization_agent done")
    return state
    

def retrieve_context(query: str) -> List[str]:
    """
    Retrieves relevant policy context using FAISS vector similarity search.
    Creates FAISS index on first run and loads from disk on subsequent runs.

    Args:
        query: Transcript text to search against policy documents

    Returns:
        List of relevant policy document chunks

    Raises:
        Exception: If policy document loading or FAISS search fails
    """
    # LangChain creates these two specific files
    faiss_file = os.path.join(POLICY_DOCS_PATH, "index.faiss")
    pkl_file = os.path.join(POLICY_DOCS_PATH, "index.pkl")

    if not os.path.isfile(faiss_file) or not os.path.isfile(pkl_file):
        log_step("retrieve_context", {"status": "index not found — creating new index"})

        # This creates a list containing ONE large Document object
        loader = TextLoader(POLICY_FILE)
        raw_documents = loader.load()

        # chunk creation
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,    # Maximum characters per chunk
            chunk_overlap=CHUNK_OVERLAP
        )
        # This creates the 'docs' list (a list of multiple Document objects)
        docs_index = text_splitter.split_documents(raw_documents)

        # initialize and create the index
        vector_store = FAISS.from_documents(docs_index, OpenAIEmbeddings())
        vector_store.save_local(POLICY_DOCS_PATH)
        log_step("retrieve_context", {"status": "index created and saved"})

    log_step("retrieve_context", {"status": "loading index"})

    db = FAISS.load_local(
        POLICY_DOCS_PATH,
        OpenAIEmbeddings(), 
        allow_dangerous_deserialization=True
    )

    docs = db.similarity_search(query, k=TOP_K_RESULTS)
    log_step("retrieve_context", {
        "status": "complete",
        "chunks_retrieved": len(docs)
    })

    return [doc.page_content for doc in docs]
