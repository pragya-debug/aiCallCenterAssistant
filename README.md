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


🧩 Architecture Overview

Pipeline flow:
```
Audio File
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
   └── complete → End Workflow (Structured Call Evaluation)

Key components
Component	        Purpose
Audio Input	        Call recording ingestion
Transcription Agent	Converts speech to text
Summarization Agent	Generates structured call summary
Policy Retriever	Retrieves QA policies via vector similarity
QA Scoring Agent	Evaluates call against policies
Routing Agent           Controls conditional flow between agents
```

🗂 Project Structure
```
aiCallCenterAssistant_ver1/
│
├── agents/
│   ├── intake_agent.py
    |-- transcription_agent.py
│   ├── summarization_agent.py
│   ├── qa_agent.py
│   └── routing_agent.py
│
├── utils/
│   └── agent_graph.py
│   └── callstate.py
│   └── check_audio.py
│   └── validation.py
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
  - Summary generated	     run QA scoring
  - QA score generated	     end workflow
  ```

⚙️ Installation
1. Clone the repository
  - git clone https://github.com/pragya-debug/aiCallCenterAssistant.git
  - cd aiCallCenterAssistant

2. Set the environment variables OPENAI_API_KEY=<your-openaikey>, KMP_DUPLICATE_LIB_OK=TRUE

3. Install all the dependencies.


🖥 Streamlit UI
To run the interactive UI:
```
streamlit run ui/streamlit_app.py

  UI Features:
  - Upload call audio
  - View transcript
  - View summary
  - View QA score
  - Visualize agent workflow
```
Sample audio files are available for testing at aiCallCenterAssistant/data/sample_transcripts
NOTE: Errors such as ```.. multiple copies of the OpenMP runtime have been linked ..```,
set env variable KMP_DUPLICATE_LIB_OK=TRUE


🚀 Future Improvements
  Potential enhancements:

  - Redis-based workflow memory
  - Call format recommendation
  - Agent feedback loops
  - Analytics dashboard


🛠 Technologies Used
  - Technology	              Role
  - Python	        Core implementation
  - LangGraph	        Agent orchestration
  - FAISS	        Policy vector retrieval
  - Streamlit	        UI
  - LLM APIs	        Summarization and QA
