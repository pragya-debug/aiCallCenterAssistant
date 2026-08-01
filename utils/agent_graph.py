from langgraph.graph import StateGraph

from agents.evaluation_agent import evaluation_agent
from agents.intake_agent import intake_agent
from agents.qa_agent import qa_agent
from agents.recommendation_agent import recommendation_agent
from agents.routing_agent import routing_agent
from agents.summarization_agent import summarization_agent
from agents.transcription_agent import transcription_agent
from utils.callstate import CallState

def build_graph():
    """
    Builds and compiles the LangGraph pipeline.
    Defines all agent nodes, edges, and conditional routing.

    Returns:
        Compiled LangGraph pipeline ready for invocation
    """

    graph = StateGraph(CallState)

    # Register all pipeline agents as nodes
    graph.add_node("intake_agent", intake_agent)
    graph.add_node("transcription_agent", transcription_agent)
    graph.add_node("summarization_agent", summarization_agent)
    graph.add_node("qa_agent", qa_agent)
    graph.add_node("recommendation_agent", recommendation_agent)
    graph.add_node("evaluation_agent", evaluation_agent)

    # Define entry point
    graph.set_entry_point("intake_agent")

    # Define fixed edges
    graph.add_edge("intake_agent", "transcription_agent")
    graph.add_edge("transcription_agent", "summarization_agent")
    graph.add_edge("summarization_agent", "qa_agent")
    graph.add_edge("recommendation_agent", "evaluation_agent")
    graph.add_edge("evaluation_agent", "__end__")

    # Conditional routing from qa_agent
    # Routing agent handles retries and determines next step
    # __end__ routes to evaluation_agent — eval runs before truly ending
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
