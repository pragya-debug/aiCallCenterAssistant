📞 AI Call Center Assistant

A modular LLM-powered call analysis pipeline that processes customer support audio calls and produces structured QA evaluation reports.

 The system performs:

 - 🎧 Call audio ingestion
 - 📝 Speech-to-text transcription
 - 📄 LLM-based call summarization
 - 📊 QA scoring against company policies
 - 🔎 Policy retrieval using vector search
 - 🤖 Agent orchestration using LangGraph
 - 🔀 Routing Agents and fallback mechanism
 - 🤖 Transcript Recommendation on poor quality


✅ Access the Assistant UI via Amazon's AWS EC2
  ```
  http: //3.134.101.96:8501  v2
  http://3.142.201.248:8501  v1
  ```
  You can download sample audio files from the UI to use.


🧩 Architecture Overview

Pipeline flow:
```
Audio File
   │
   ▼
Intake Agent   
   │
   ▼
Transcription Agent
   │
   ▼
Summarization Agent
   │
   ▼
Policy Retrieval (FAISS)
   │
   ▼
QA Scoring Agent
   │
   ▼
Routing Agent
   │
   ├── retry_transcription → Transcription Agent
   ├── retry_summary → Summarization Agent
   ├── retry_qa → QA Scoring Agent
   ├── recommendation → Recommendation Agent (if QA score <=50%)
   └── complete → End Workflow (Structured Call Evaluation)

Key components
Component	        Purpose
Audio Input	        Call recording ingestion
Transcription Agent	Converts speech to text
Summarization Agent	Generates structured call summary
Policy Retriever	Retrieves QA policies via vector similarity
QA Scoring Agent	Evaluates call against policies
Routing Agent           Controls conditional flow between agents
Execution Tracing       Trace the agents called during execution
Recommendation Agent	Recommends improved steps/transcript to enhance Quality
```


🗂 Project Structure
```
aiCallCenterAssistant/
│
├── agents/
│   ├── intake_agent.py
│   ├── transcription_agent.py
│   ├── summarization_agent.py
│   ├── qa_agent.py
│   └── routing_agent.py
│   └── recommendation_agent.py
│
├── utils/
│   └── agent_graph.py
│   └── callstate.py
│   └── check_audio.py
│   └── validation.py
│   └── logger.py
│
├── data/
│   ├── policy_docs/
│   └── sample_transcripts/
│
├── ui/
│   └── streamlit_app.py
│
└── README.md
```


🧠 Workflow Orchestration

The pipeline is implemented using LangGraph which manages agent execution and state transitions.


🧾 State Model

The workflow state is defined as a typed dictionary.
  ```
  from typing import TypedDict, Optional, Dict
  class CallState(TypedDict, total=False):
    audio_path: str
    transcript: Optional[str]
    summary: Optional[Dict]
    qa_score: Optional[Dict]
    recommendation: Optional[Dict]
    improved_transcript: str

    error: Optional[str]
    retry_count: int
    trace: list[str]
    next: str
  ```


🔎 Policy Retrieval
  - QA policies are stored as embeddings using FAISS.
  - The system retrieves the most relevant policies before performing QA scoring.


🤖 Routing Agent

The routing agent determines workflow transitions based on model output.
  Examples:
  ```
      Condition	                Next Step
  - Transcript empty	     retry transcription
  - Transcript generated     run summarization
  - Summarization empty	     retry summarization
  - Summary generated	     run QA scoring
  - QA score empty	     retry QA Scoring
  - QA score generated	     if score <= 50%, ask recommendation else, end workflow
  - Recommendation created   end workflow
  ```


⚙️ Installation

Local (tested on mac):
1. Clone the repository
  - git clone https://github.com/pragya-debug/aiCallCenterAssistant.git
  - cd aiCallCenterAssistant

2. Set the environment variables OPENAI_API_KEY=<your-openaikey>, KMP_DUPLICATE_LIB_OK=TRUE

3. Install all the dependencies
   ``` 
   streamlit langgraph langchain faiss-cpu openai whisper dotenv
   langchain-community langchain-openai ffmpeg
   ```


🖥 Streamlit UI

To run the interactive UI:
  ```
  Locally:
  cd aiCallCenterAssistant
  streamlit run ui/streamlit_app.py

  UI Features:
  - Upload call audio
  - View transcript
  - View summary
  - View QA score
  - Visualize agent workflow
  - Execution Trace
  - Recommendation of quality improvement and improved transcript
  ```
Sample audio files are available for testing at aiCallCenterAssistant/data/sample_transcripts
NOTE: Errors such as ```.. multiple copies of the OpenMP runtime have been linked ..```,
set env variable KMP_DUPLICATE_LIB_OK=TRUE


🚀 Future Improvements

  - Redis-based workflow memory
  - Call format recommendation (in V2)
  - Agent feedback loops (in V2)
  - Analytics dashboard


🛠 Technologies Used
  ```
  - Technology	              Role
  - Python              Core implementation
  - LangGraph           Agent orchestration
  - FAISS               Policy vector retrieval
  - Streamlit           UI
  - LLM APIs            Summarization and QA
  ```
