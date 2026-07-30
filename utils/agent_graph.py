from langgraph.graph import StateGraph

from agents.intake_agent import intake_agent
from agents.transcription_agent import transcription_agent
from agents.summarization_agent import summarization_agent
from agents.recommendation_agent import recommendation_agent
from agents.qa_agent import qa_agent
from agents.routing_agent import routing_agent
from agents.evaluation_agent import evaluation_agent
from utils.callstate import CallState
from utils.logger import log_step

def build_graph():

    graph = StateGraph(CallState)

    graph.add_node("intake_agent", intake_agent)
    graph.add_node("transcription_agent", transcription_agent)
    graph.add_node("summarization_agent", summarization_agent)
    graph.add_node("qa_agent", qa_agent)
    graph.add_node("recommendation_agent", recommendation_agent)
    graph.add_node("evaluation_agent", evaluation_agent)

    graph.set_entry_point("intake_agent")

    graph.add_edge("intake_agent", "transcription_agent")
    graph.add_edge("transcription_agent", "summarization_agent")
    graph.add_edge("summarization_agent", "qa_agent")
    graph.add_edge("recommendation_agent", "evaluation_agent")
    graph.add_edge("evaluation_agent", "__end__")
    graph.add_conditional_edges("qa_agent",
                                routing_agent,
                                {
                                    "transcription_agent": "transcription_agent",
                                    "summarization_agent": "summarization_agent",
                                    "qa_agent": "qa_agent",
                                    "recommendation_agent": "recommendation_agent",
                                    "__end__": "evaluation_agent" #eval runs before truly ending
                                }
    )

    return graph.compile()
