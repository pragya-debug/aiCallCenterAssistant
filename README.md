📞 AI Call Center Assistant

Multi-agent AI system that converts call center audio into structured insights, QA scores, and coaching recommendations using LLMs, RAG, and LangGraph.

 The system performs:

 - 🎧 Call audio ingestion
 - 📝 Speech-to-text transcription
 - 📄 LLM-based call summarization
 - 📊 QA scoring against company policies
 - 🔎 Policy retrieval using vector search
 - 🤖 Agent orchestration using LangGraph
 - 🔀 Routing Agents and fallback mechanism
 - 🤖 Transcript Recommendation on poor quality
 - 🛡️ Evaluation framework with automated quality and safety checks


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
   └── recommendation → Recommendation Agent (if QA score <= 50%)
                        │
              (all paths converge)
                        │
                        ▼
                Evaluation Agent 
                        │
                        ▼
               Evaluation Framework
                        │
                        ▼
                Evaluation Report
                        │
                        ▼
                     __end__ (End Workflow)

Key components
Component               Purpose
Audio Input             Call recording ingestion
Transcription Agent     Converts speech to text
Summarization Agent     Generates structured call summary
Policy Retriever        Retrieves QA policies via vector similarity
QA Scoring Agent        Evaluates call against policies
Routing Agent           Controls conditional flow between agents
Execution Tracing       Trace the agents called during execution
Recommendation Agent    Recommends improved steps/transcript to enhance Quality
Evaluation Framework    Validates pipeline outputs across multi dimensions
Evaluation Report       Structured JSON report with pass rates and per-dimension results
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
│   ├── evaluation_agent.py
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
├── tests/
│   └── test_evaluate.py
│
├── evaluate.py
└── README.md
```


🧠 Workflow Orchestration

The pipeline is implemented using LangGraph which manages agent execution and state transitions.


🧾 State Model

The workflow state is defined as a typed dictionary.
  ```
  from typing import TypedDict, Optional, Dict
  class CallState(TypedDict, total=False):
    # Pipeline inputs
    audio_path: str

    # Agent outputs
    transcript: Optional[str]
    summary: Optional[Dict]
    qa_score: Optional[Dict]
    recommendation: Optional[Dict]
    improved_transcript: str

    # Flow control
    error: Optional[str]
    retry_count: int
    trace: list[str]
    next: str

    # Evaluation outputs
    eval_report: Optional[Dict]
    eval_pass_rate: Optional[float]
    eval_passed: Optional[int]
    eval_total: Optional[int]
  ```


🔎 Policy Retrieval
  - QA policies are stored as embeddings using FAISS.
  - The system retrieves the most relevant policies before performing QA scoring.


🤖 Routing Agent

The routing agent determines workflow transitions based on model output.
  Examples:
  ```
      Condition	                Next Step
  - Transcript empty         retry Transcription Agent
  - Transcript generated     run Summarization Agent
  - Summarization empty      retry Summarization Agent
  - Summary generated        run QA Scoring Agent
  - QA score empty           retry QA Scoring Agent
  - QA score <= 50%          run Recommendation Agent
  - QA score > 50%           run Evaluation Agent
  - Recommendation created   run Evaluation Agent
  - Evaluation run           end workflow
  ```


⚙️  Installation

Local (tested on mac):
1. Clone the repository
  ```
  git clone https://github.com/pragya-debug/aiCallCenterAssistant.git
  cd aiCallCenterAssistant
  ```

2. Set the environment variables
   Create a `.env` file in the root folder and add keys:
   ```
   touch .env
   OPENAI_API_KEY=your_openai_api_key_here
   KMP_DUPLICATE_LIB_OK=TRUE
   ```

3. Install all the dependencies
   ``` 
   streamlit langgraph langchain faiss-cpu openai whisper python-dotenv
   langchain-community langchain-openai ffmpeg pytest
   ```
🖥 Streamlit UI

To run the interactive UI locally:
  ```
  cd aiCallCenterAssistant
  streamlit run ui/streamlit_app.py

  Available for live demo on request.

  UI Features:
  - Upload call audio
  - View transcript
  - View summary
  - View QA score
  - Visualize agent workflow
  - Execution Trace 
  - Recommendation of quality improvement and improved transcript
  - Evaluation results
  ```
Sample audio files are available for testing at aiCallCenterAssistant/data/sample_transcripts


🛡️ Evaluation Framework

CallSense includes a built-in evaluation framework that automatically validates pipeline outputs across five safety and quality dimensions.

  - Why Evaluation Matters
  AI systems produce probabilistic outputs — unlike traditional software, the same input can produce varying results. 
  The evaluation framework catches quality issues, hallucinations, and safety gaps before they reach end users.

  - Five Evaluation Dimensions

  | Dimension | What It Checks | Pass Criteria |
  |-----------|---------------|---------------|
  | Transcription Completeness | Transcript meets minimum length — short transcript indicates transcription failure | > 50 characters |
  | Summary Faithfulness | Summary words are grounded in transcript — detects hallucination | > 40% word overlap |
  | QA Score Validity | Quality score is within valid range — invalid score indicates scoring agent failure | 0.0 – 1.0 |
  | Routing Logic | Correct agent triggered based on QA score — low scores must route to recommendation agent | Score-based routing |
  | Recommendation Presence | Coaching recommendation present for low-scoring calls — missing recommendation is a safety gap | Required when QA < 0.5 |

  - Running Evaluations

  ```bash
  # Run evaluation suite against sample outputs
  python evaluate.py

  # Run full unit test suite — 44 tests
  python -m pytest tests/test_evaluate.py -v
  ```

  - Sample Evaluation Report

  ```json
  {
    "timestamp": "2026-07-25T10:00:00",
    "pass_rate": 1.0,
    "passed": 5,
    "total": 5,
    "results": [
      {"test_name": "transcription_completeness", "passed": true, "score": 1.0},
      {"test_name": "summary_faithfulness", "passed": true, "score": 0.82},
      {"test_name": "qa_score_validity", "passed": true, "score": 1.0},
      {"test_name": "routing_logic", "passed": true, "score": 1.0},
      {"test_name": "recommendation_presence", "passed": true, "score": 1.0}
    ]
  }
  ```

  - Test Coverage
    - 44 pytest unit tests covering pass cases, fail cases, edge cases, and boundary conditions
    - Tests organized by evaluation dimension
    - All 44 tests passing


🚀 Future Improvements

  - Bug Fixes & Pipeline Improvements
    - Fix summary faithfulness — improve summarization agent grounding to reduce hallucination,
    - Fix routing state — preserve next agent field through evaluation step

  - Evaluation Enhancements
    - LLM as judge — use second LLM to evaluate output quality beyond word overlap
    - PII redaction — detect and remove personally identifiable information from transcripts

  - Platform Enhancements
    - Redis-based workflow memory
    - Analytics dashboard


🛠 Technologies Used
  ```
  - Technology	              Role
  - Python              Core implementation
  - LangGraph           Agent orchestration and pipeline management
  - LangChain           RAG pipeline utilities
  - FAISS               Policy vector retrieval
  - Streamlit           User Interface
  - Whisper             Speech-to-text transcription
  - OpenAI GPT-4o       LLM for summarization, QA scoring, and recommendations
  - pytest              Eval framework unit testing
  ```
