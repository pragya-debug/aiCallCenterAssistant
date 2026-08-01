"""
AI CallSense — Evaluation Framework Unit Tests
Tests all five evaluation dimensions with pass and fail cases.
"""

import pytest
from evaluate import (
    evaluate_transcription_completeness,
    evaluate_summary_faithfulness,
    evaluate_qa_score_validity,
    evaluate_routing_logic,
    evaluate_recommendation_presence,
    run_evaluation_suite
)


# ── TRANSCRIPTION COMPLETENESS TESTS ──

class TestTranscriptionCompleteness:
    """Tests for evaluate_transcription_completeness function."""

    def test_long_transcript_passes(self):
        transcript = "Customer called about billing issue. " * 5
        result = evaluate_transcription_completeness(transcript)
        assert result["passed"] is True
        assert result["score"] == 1.0

    def test_empty_transcript_fails(self):
        result = evaluate_transcription_completeness("")
        assert result["passed"] is False
        assert result["score"] == 0.0

    def test_short_transcript_fails(self):
        result = evaluate_transcription_completeness("too short")
        assert result["passed"] is False

    def test_whitespace_only_fails(self):
        result = evaluate_transcription_completeness("   ")
        assert result["passed"] is False

    def test_exactly_minimum_length_passes(self):
        transcript = "x" * 50
        result = evaluate_transcription_completeness(transcript, min_length=50)
        assert result["passed"] is True

    def test_one_below_minimum_fails(self):
        transcript = "x" * 49
        result = evaluate_transcription_completeness(transcript, min_length=50)
        assert result["passed"] is False

    def test_custom_minimum_length(self):
        transcript = "x" * 100
        result = evaluate_transcription_completeness(transcript, min_length=200)
        assert result["passed"] is False


# ── SUMMARY FAITHFULNESS TESTS ──

class TestSummaryFaithfulness:
    """Tests for evaluate_summary_faithfulness function."""

    def test_faithful_summary_passes(self):
        transcript = "Customer called about billing charges on their account refund requested"
        summary = "Customer called about billing charges refund requested"
        result = evaluate_summary_faithfulness(summary, transcript)
        assert result["passed"] is True

    def test_hallucinated_summary_fails(self):
        transcript = "Customer called about billing"
        summary = "Customer complained about shipping delivery tracking package lost"
        result = evaluate_summary_faithfulness(summary, transcript)
        assert result["passed"] is False

    def test_empty_summary_fails(self):
        result = evaluate_summary_faithfulness("", "some transcript content here")
        assert result["passed"] is False
        assert result["score"] == 0.0

    def test_empty_transcript_fails(self):
        result = evaluate_summary_faithfulness("some summary", "")
        assert result["passed"] is False

    def test_both_empty_fails(self):
        result = evaluate_summary_faithfulness("", "")
        assert result["passed"] is False

    def test_score_is_between_zero_and_one(self):
        transcript = "Customer called about refund billing account charges"
        summary = "Customer called about refund billing"
        result = evaluate_summary_faithfulness(summary, transcript)
        assert 0.0 <= result["score"] <= 1.0

    def test_custom_threshold(self):
        transcript = "Customer called about billing"
        summary = "Customer called"
        result = evaluate_summary_faithfulness(summary, transcript, threshold=0.1)
        assert result["passed"] is True


# ── QA SCORE VALIDITY TESTS ──

class TestQAScoreValidity:
    """Tests for evaluate_qa_score_validity function."""

    def test_valid_mid_score_passes(self):
        result = evaluate_qa_score_validity(0.75)
        assert result["passed"] is True
        assert result["score"] == 1.0

    def test_zero_score_passes(self):
        result = evaluate_qa_score_validity(0.0)
        assert result["passed"] is True

    def test_one_score_passes(self):
        result = evaluate_qa_score_validity(1.0)
        assert result["passed"] is True

    def test_above_one_fails(self):
        result = evaluate_qa_score_validity(1.5)
        assert result["passed"] is False
        assert result["score"] == 0.0

    def test_negative_score_fails(self):
        result = evaluate_qa_score_validity(-0.1)
        assert result["passed"] is False

    def test_none_score_fails(self):
        result = evaluate_qa_score_validity(None)
        assert result["passed"] is False

    def test_string_score_fails(self):
        result = evaluate_qa_score_validity("high")
        assert result["passed"] is False

    def test_very_high_score_fails(self):
        result = evaluate_qa_score_validity(100)
        assert result["passed"] is False


# ── ROUTING LOGIC TESTS ──

class TestRoutingLogic:
    """Tests for evaluate_routing_logic function."""

    def test_low_score_routes_to_recommendation(self):
        result = evaluate_routing_logic(0.45, "recommendation")
        assert result["passed"] is True

    def test_high_score_routes_to_end(self):
        result = evaluate_routing_logic(0.75, "end")
        assert result["passed"] is True

    def test_exactly_at_threshold_routes_to_end(self):
        result = evaluate_routing_logic(0.5, "end")
        assert result["passed"] is True

    def test_just_below_threshold_routes_to_recommendation(self):
        result = evaluate_routing_logic(0.49, "recommendation")
        assert result["passed"] is True

    def test_low_score_wrong_routing_fails(self):
        result = evaluate_routing_logic(0.3, "end")
        assert result["passed"] is False
        assert result["score"] == 0.0

    def test_high_score_wrong_routing_fails(self):
        result = evaluate_routing_logic(0.8, "recommendation")
        assert result["passed"] is False

    def test_invalid_qa_score_fails(self):
        result = evaluate_routing_logic("invalid", "end")
        assert result["passed"] is False

    def test_custom_threshold(self):
        result = evaluate_routing_logic(0.6, "recommendation", threshold=0.7)
        assert result["passed"] is True


# ── RECOMMENDATION PRESENCE TESTS ──

class TestRecommendationPresence:
    """Tests for evaluate_recommendation_presence function."""

    def test_low_score_with_recommendation_passes(self):
        result = evaluate_recommendation_presence(
            0.35, "Follow up with customer within 24 hours"
        )
        assert result["passed"] is True

    def test_low_score_without_recommendation_fails(self):
        result = evaluate_recommendation_presence(0.35, "")
        assert result["passed"] is False
        assert result["score"] == 0.0

    def test_high_score_without_recommendation_passes(self):
        result = evaluate_recommendation_presence(0.85, "")
        assert result["passed"] is True

    def test_high_score_with_recommendation_passes(self):
        result = evaluate_recommendation_presence(0.85, "Optional follow up")
        assert result["passed"] is True

    def test_exactly_at_threshold_no_recommendation_passes(self):
        result = evaluate_recommendation_presence(0.5, "")
        assert result["passed"] is True

    def test_just_below_threshold_needs_recommendation(self):
        result = evaluate_recommendation_presence(0.49, "")
        assert result["passed"] is False

    def test_whitespace_recommendation_fails_for_low_score(self):
        result = evaluate_recommendation_presence(0.3, "   ")
        assert result["passed"] is False

    def test_invalid_score_fails(self):
        result = evaluate_recommendation_presence("bad", "recommendation text")
        assert result["passed"] is False


# ── FULL SUITE TESTS ──

class TestEvaluationSuite:
    """Integration tests for run_evaluation_suite function."""

    def test_good_output_all_pass(self):
        good_output = {
            "transcript": "Customer called about billing issue on account. " * 3,
            "summary": "Customer called about billing issue on account.",
            "qa_score": 0.85,
            "next_agent": "end",
            "recommendation": ""
        }
        report = run_evaluation_suite(good_output)
        assert report["pass_rate"] == 1.0
        assert report["passed"] == 5
        assert report["total"] == 5

    def test_low_score_output_with_recommendation_passes(self):
        output = {
            "transcript": "Customer called about refund for damaged product order number. " * 2,
            "summary": "Customer called about refund for damaged product.",
            "qa_score": 0.35,
            "next_agent": "recommendation",
            "recommendation": "Follow up with customer about refund status."
        }
        report = run_evaluation_suite(output)
        assert report["pass_rate"] == 1.0

    def test_failure_output_has_failures(self):
        failure_output = {
            "transcript": "short",
            "summary": "completely unrelated weather sports content",
            "qa_score": 1.5,
            "next_agent": "end",
            "recommendation": ""
        }
        report = run_evaluation_suite(failure_output)
        assert report["pass_rate"] < 1.0
        assert report["passed"] < report["total"]

    def test_report_has_required_fields(self):
        output = {
            "transcript": "Customer called about issue. " * 5,
            "summary": "Customer called about issue.",
            "qa_score": 0.7,
            "next_agent": "end",
            "recommendation": ""
        }
        report = run_evaluation_suite(output)
        assert "timestamp" in report
        assert "pass_rate" in report
        assert "passed" in report
        assert "total" in report
        assert "results" in report
        assert len(report["results"]) == 5

    def test_missing_fields_handled_gracefully(self):
        empty_output = {}
        report = run_evaluation_suite(empty_output)
        assert "pass_rate" in report
        assert report["total"] == 5

    def test_pass_rate_between_zero_and_one(self):
        output = {
            "transcript": "x" * 100,
            "summary": "customer called",
            "qa_score": 0.6,
            "next_agent": "end",
            "recommendation": ""
        }
        report = run_evaluation_suite(output)
        assert 0.0 <= report["pass_rate"] <= 1.0
