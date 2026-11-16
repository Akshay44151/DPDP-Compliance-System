import json
from pathlib import Path

from agents.implementation_agent.agent import ImplementationExtractionAgent
from agents.implementation_agent.scanner import scan_project

APPS = Path(__file__).parent / "fixtures" / "apps"


def test_scanner_flags_violating_app():
    ev, files, cov = scan_project(APPS / "acmenotes_violating")
    assert files == 3
    assert cov["model_field"] >= 5          # email, name, phone, ip, aadhaar, dob
    assert cov["data_write"] == 1
    assert cov["outbound_email"] == 1
    assert cov["data_sharing"] == 1         # ads tracker
    assert cov["personal_data_logged"] == 1
    for missing in ("consent_logic", "consent_ui", "privacy_notice_link",
                    "retention_logic", "retention_field"):
        assert missing not in cov           # absence is visible
    mail = next(e for e in ev if e.kind == "outbound_email")
    assert mail.detail["references_consent"] is False


def test_scanner_recognises_baseline_controls():
    ev, _, cov = scan_project(APPS / "acmenotes_baseline")
    assert cov["consent_ui"] == 1
    assert cov["privacy_notice_link"] == 1
    assert cov["retention_logic"] >= 1      # purge_inactive_accounts command
    assert cov["retention_field"] == 1
    assert "data_sharing" not in cov
    mail = next(e for e in ev if e.kind == "outbound_email")
    assert mail.detail["references_consent"] is True


class FakeLLM:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    def generate(self, *, system_prompt, user_prompt, response_schema=None):
        self.calls += 1
        return self.replies.pop(0)


def test_agent_drops_hallucinated_ids_and_retries():
    good = json.dumps({"facts": [{"statement": "Profile data is written without a consent check.",
                                  "fact_type": "partial", "evidence_ids": ["E001", "E999"],
                                  "confidence": 0.9}]})
    llm = FakeLLM(["not json"] + [good] * 10)
    report = ImplementationExtractionAgent(llm).run(APPS / "acmenotes_violating")
    assert llm.calls == 6                    # 5 categories + 1 retry
    assert all("E999" not in f.evidence_ids for f in report.facts)
    assert report.facts[0].locations
    assert {f.category.value for f in report.facts} == {
        "transparency_notice", "consent", "purpose_limitation",
        "data_minimization", "storage_retention"}
