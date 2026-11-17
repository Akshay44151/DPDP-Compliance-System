"""Agent 2 reports which categories' LLM calls gave up (ImplementationReport.failed_categories)."""
from pathlib import Path

from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableLambda

from agents.implementation_agent.agent import CATEGORY_KINDS, ImplementationExtractionAgent
from agents.implementation_agent.schemas import ImplementationReport


class _FailingModel:
    """Every structured-output call raises a parse error, so every category fails."""

    def with_structured_output(self, schema):
        def boom(_):
            raise OutputParserException("bad output")
        return RunnableLambda(boom)


def test_failed_categories_are_listed():
    app = Path(__file__).parent / "fixtures" / "apps" / "acmenotes_baseline"
    agent = ImplementationExtractionAgent(model=_FailingModel())
    report = agent.run(app)
    assert report.llm_failures == len(CATEGORY_KINDS)
    assert report.failed_categories == [c.value for c in CATEGORY_KINDS]
    assert report.facts == []


def test_failed_categories_resets_between_runs():
    app = Path(__file__).parent / "fixtures" / "apps" / "acmenotes_baseline"
    agent = ImplementationExtractionAgent(model=_FailingModel())
    agent.run(app)
    second = agent.run(app)
    assert len(second.failed_categories) == len(CATEGORY_KINDS)  # not doubled


def test_failed_categories_defaults_to_empty():
    report = ImplementationReport(project_path="x", files_scanned=0, coverage={},
                                  evidence=[], facts=[])
    assert report.failed_categories == []