from agents.policy_agent.agent import PolicyExtractionAgent
from agents.policy_agent.schemas import PolicyExtractionResult
from agents.policy_agent.verification import ExtractionReport
from orchestrator.interfaces import PolicyExtractor


class FakeExtractor:
    def extract_with_report(self, *, document_text: str, source: str):
        return ExtractionReport(result=PolicyExtractionResult(claims=[]))


class NotAnExtractor:
    def something_else(self):
        return None


def test_real_agent_1_satisfies_the_contract():
    # Checked on the class, so no model is built and no API is called.
    assert issubclass(PolicyExtractionAgent, PolicyExtractor)


def test_a_simple_fake_satisfies_the_contract():
    assert isinstance(FakeExtractor(), PolicyExtractor)


def test_an_unrelated_object_does_not_satisfy_the_contract():
    assert not isinstance(NotAnExtractor(), PolicyExtractor)