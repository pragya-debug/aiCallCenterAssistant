"""
AI CallSense — Evaluation Framework
Evaluates pipeline outputs across five safety and quality dimensions.
"""

from typing import TypedDict
import json
from datetime import datetime

class EvalResult(TypedDict):
    """Single evaluation dimension result."""
    test_name: str
    passed: bool
    score: float
    details: str

class EvalReport(TypedDict):
    """Complete evaluation report across all dimensions."""
    timestamp: str
    pass_rate: float
    passed: int
    total: int
    results: list


# EVAL 1 - Transcription Completeness - 

def evaluate_transcription_completeness(
    transcript: str,
    min_length: int = 50
) -> EvalResult:
    """
    Check transcript meets minimum length requirement.
    A very short transcript likely indicates a transcription failure.
    """
    transcript = transcript.strip() if transcript else ""
    passed = len(transcript) >= min_length

    return {
	"test_name": "transcription_completeness",
	"passed": passed,
        "score": 1.0 if passed else 0.0,
        "details": (
            f"Transcript length: {len(transcript)} chars "
            f"(minimum: {min_length})"
        )
    }


# EVAL 2 - Summary Faithfullness -

def evaluate_summary_faithfulness(
    summary: str,
    transcript: str,
    threshold: float = 0.4
) -> EvalResult:
    """
    Check summary words are grounded in transcript content.
    Detects hallucination — summary claims not present in source.
    """
    summary = summary.strip().lower() if summary else ""
    transcript = transcript.strip().lower() if transcript else ""

    if not summary:
        return {
            "test_name": "summary_faithfulness",
            "passed": False,
            "score": 0.0,
            "details": "Empty summary — faithfulness cannot be evaluated"
        }

    if not transcript:
        return {
            "test_name": "summary_faithfulness",
            "passed": False,
            "score": 0.0,
            "details": "Empty transcript — cannot evaluate faithfulness"
        }

    # Remove common stop words for cleaner overlap
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on",
        "at", "to", "for", "of", "with", "is", "was", "are",
        "were", "be", "been", "have", "has", "had", "do", "did",
        "will", "would", "could", "should", "may", "might", "i",
        "we", "you", "he", "she", "they", "it", "this", "that"
    }

    summary_words = set(summary.split()) - stop_words
    transcript_words = set(transcript.split()) - stop_words

    if not summary_words:
        return {
            "test_name": "summary_faithfulness",
            "passed": False,
            "score": 0.0,
            "details": "Summary contains only stop words"
        }

    overlap = summary_words & transcript_words
    score = len(overlap) / len(summary_words)
    passed = score >= threshold

    return {
        "test_name": "summary_faithfulness",
        "passed": passed,
        "score": round(score, 2),
        "details": (
            f"Word overlap: {score:.0%} "
            f"({len(overlap)}/{len(summary_words)} meaningful words) "
            f"(threshold: {threshold:.0%})"
        )
    }


# EVAL 3 - QA Score Validity -

def evaluate_qa_score_validity(
    qa_score: float
) -> EvalResult:
    """
    Check QA score is within valid range 0.0 to 1.0.
    Invalid scores indicate a scoring agent failure.
    """
    try:
        score_float = float(qa_score)
        passed = 0.0 <= score_float <= 1.0
    except (TypeError, ValueError):
        return {
            "test_name": "qa_score_validity",
            "passed": False,
            "score": 0.0,
            "details": f"QA score is not a valid number: {qa_score}"
        }

    return {
        "test_name": "qa_score_validity",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "details": (
            f"QA score: {score_float} "
            f"({'valid' if passed else 'out of range — must be 0.0 to 1.0'})"
        )
    }


# EVAL 4 - Routing Logic Correctness -

def evaluate_routing_logic(
    qa_score: float,
    next_agent: str,
    threshold: float = 0.5
) -> EvalResult:
    """
    Check routing agent correctly routes based on QA score.
    Below threshold should trigger recommendation agent.
    Above threshold should terminate pipeline.
    """
    try:
        score_float = float(qa_score)
    except (TypeError, ValueError):
        return {
            "test_name": "routing_logic",
            "passed": False,
            "score": 0.0,
            "details": f"Cannot evaluate routing — invalid QA score: {qa_score}"
        }

    expected = "recommendation" if score_float < threshold else "end"
    passed = next_agent == expected

    return {
        "test_name": "routing_logic",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "details": (
            f"QA score {score_float} → "
            f"expected agent: '{expected}', "
            f"actual agent: '{next_agent}' "
            f"({'correct' if passed else 'INCORRECT ROUTING'})"
        )
    }


# EVAL 5 - Recommendation Present When Needed -

def evaluate_recommendation_presence(
    qa_score: float,
    recommendation: str,
    threshold: float = 0.5
) -> EvalResult:
    """
    Check recommendation is present for low-scoring calls.
    Low QA score without recommendation is a safety gap.
    """
    try:
        score_float = float(qa_score)
    except (TypeError, ValueError):
        return {
            "test_name": "recommendation_presence",
            "passed": False,
            "score": 0.0,
            "details": f"Cannot evaluate — invalid QA score: {qa_score}"
        }

    recommendation = recommendation.strip() if recommendation else ""
    recommendation_needed = score_float < threshold

    if recommendation_needed:
        passed = len(recommendation) > 0
        details = (
            f"QA score {score_float} below threshold {threshold} — "
            f"recommendation {'present' if passed else 'MISSING'}"
        )
    else:
        passed = True
        details = (
            f"QA score {score_float} above threshold {threshold} — "
            f"recommendation not required"
        )

    return {
        "test_name": "recommendation_presence",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "details": details
    }

# ── EVALUATION SUITE ──

def run_evaluation_suite(pipeline_output: dict) -> EvalReport:
    """
    Run all five evaluations against pipeline output.
    Returns structured report with pass rate and detailed results.

    Expected pipeline_output keys:
        transcript (str): Full call transcript
        summary (str): Agent-generated summary
        qa_score (float): Quality score 0.0-1.0
        next_agent (str): Agent routing decision
        recommendation (str): Coaching recommendation if needed
    """
    results = []

    results.append(evaluate_transcription_completeness(
        pipeline_output.get("transcript", "")
    ))

    results.append(evaluate_summary_faithfulness(
        summary=pipeline_output.get("summary", ""),
        transcript=pipeline_output.get("transcript", "")
    ))

    results.append(evaluate_qa_score_validity(
        pipeline_output.get("qa_score", -1)
    ))

    results.append(evaluate_routing_logic(
        qa_score=pipeline_output.get("qa_score", 0),
        next_agent=pipeline_output.get("next_agent", "")
    ))

    results.append(evaluate_recommendation_presence(
        qa_score=pipeline_output.get("qa_score", 0),
        recommendation=pipeline_output.get("recommendation", "")
    ))

    passed_count = sum(1 for r in results if r["passed"])
    total = len(results)

    return {
        "timestamp": datetime.now().isoformat(),
        "pass_rate": round(passed_count / total, 2),
        "passed": passed_count,
        "total": total,
        "results": results
    }


def print_report(report: EvalReport) -> None:
    """Print evaluation report in readable format."""
    print("\n" + "="*60)
    print("AI CALLSENSE — EVALUATION REPORT")
    print("="*60)
    print(f"Timestamp:  {report['timestamp']}")
    print(f"Pass Rate:  {report['pass_rate']:.0%} "
          f"({report['passed']}/{report['total']} passed)")
    print("-"*60)

    for result in report["results"]:
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        print(f"\n{status} — {result['test_name']}")
        print(f"  Score:   {result['score']:.2f}")
        print(f"  Details: {result['details']}")

    print("\n" + "="*60)


# ── EXAMPLE USAGE ──

if __name__ == "__main__":

    # Test Case 1 — Good pipeline output
    print("\nTest Case 1: Good pipeline output")
    good_output = {
        "transcript": (
            "Customer called about a billing issue on their account. "
            "The agent reviewed the charges and explained the breakdown. "
            "Customer was satisfied with the explanation and the agent "
            "offered a courtesy credit for the inconvenience."
        ),
        "summary": (
            "Customer called about billing charges. "
            "Agent explained breakdown and offered courtesy credit."
        ),
        "qa_score": 0.85,
        "next_agent": "end",
        "recommendation": ""
    }
    report = run_evaluation_suite(good_output)
    print_report(report)

    # Test Case 2 — Low QA score needing recommendation
    print("\nTest Case 2: Low QA score — recommendation required")
    low_score_output = {
        "transcript": (
            "Customer called about a refund request for a damaged product. "
            "The agent was unable to locate the order and placed the customer "
            "on hold multiple times without resolution."
        ),
        "summary": (
            "Customer called about refund for damaged product. "
            "Agent could not resolve the issue."
        ),
        "qa_score": 0.35,
        "next_agent": "recommendation",
        "recommendation": (
            "Follow up with customer within 24 hours. "
            "Locate order using alternate search methods. "
            "Escalate to supervisor if refund cannot be processed directly."
        )
    }
    report = run_evaluation_suite(low_score_output)
    print_report(report)

    # Test Case 3 — Failure cases
    print("\nTest Case 3: Pipeline failures")
    failure_output = {
        "transcript": "short",
        "summary": "completely unrelated content about weather and sports",
        "qa_score": 1.5,
        "next_agent": "end",
        "recommendation": ""
    }
    report = run_evaluation_suite(failure_output)
    print_report(report)

    # Save last report as JSON example
    with open("eval_report_example.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nExample report saved to eval_report_example.json")

