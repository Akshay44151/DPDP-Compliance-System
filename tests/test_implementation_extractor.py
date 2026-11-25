from pathlib import Path

from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableLambda

from agents.implementation_agent.agent import ImplementationExtractionAgent
from agents.implementation_agent.scanner import scan_project

APPS = Path(__file__).parent / "fixtures" / "apps"


def test_scanner_flags_violating_app():
    ev, files, cov = scan_project(APPS / "acmenotes_violating")
    assert files == 3
    assert cov["model_field"] >= 7          # email, name, phone, ip, device, aadhaar, dob
    assert cov["data_write"] == 1
    assert cov["outbound_email"] == 1
    assert cov["data_sharing"] == 1         # ads tracker
    assert cov["personal_data_logged"] == 1
    for missing in ("consent_logic", "consent_ui", "privacy_notice_link",
                    "retention_logic", "retention_field"):
        assert missing not in cov           # absence is visible
    mail = next(e for e in ev if e.kind == "outbound_email")
    assert mail.detail["references_consent"] is False
    assert mail.detail["subjects"] == ["Weekly offers"]
    write = next(e for e in ev if e.kind == "data_write")
    assert {"email", "aadhaar_number", "device_id"} <= set(write.detail["fields"])
    share = next(e for e in ev if e.kind == "data_sharing")
    assert "ads.example.com/track" in share.detail["target"]       # the LLM can see it is an ad tracker


def test_scanner_recognises_baseline_controls():
    ev, _, cov = scan_project(APPS / "acmenotes_baseline")
    assert cov["consent_ui"] == 2          # marketing checkbox + unsubscribe link
    assert any(e.detail.get("control") == "withdraw_link" for e in ev)
    assert cov["privacy_notice_link"] == 1
    assert cov["retention_logic"] >= 1      # purge_inactive_accounts command
    assert cov["retention_field"] == 1
    assert "data_sharing" not in cov
    mail = next(e for e in ev if e.kind == "outbound_email")
    assert mail.detail["references_consent"] is True
    assert mail.detail["subjects"] == ["Product updates"]
    # deletion code is read, not just detected by file name
    purge = next(e for e in ev if e.kind == "retention_logic" and e.detail["role"] == "deletion_code")
    assert purge.detail["period_days"] == 730
    assert "last_login__lt" in purge.detail["cutoff_filters"]
    assert next(e for e in ev if e.detail.get("role") == "file").snippet  # never an empty snippet
    assert any(e.kind == "setting" and e.detail["name"] == "BACKUP_RETENTION_DAYS" for e in ev)
    assert any(e.kind == "consent_logic" and e.detail.get("name") == "withdraw_consent" for e in ev)


class FakeChatModel:
    """Stands in for ChatGroq: supports .with_structured_output(cls) and prompt piping.

    `script(text, cls)` returns a dict (validated into cls) or an Exception (raised).
    """

    def __init__(self, script):
        self.script, self.calls = script, 0

    def with_structured_output(self, cls):
        def run(prompt_value):
            self.calls += 1
            out = self.script(prompt_value.messages[-1].content, cls)
            if isinstance(out, Exception):
                raise out
            return cls.model_validate(out)
        return RunnableLambda(run)


def queue(replies):
    replies = list(replies)
    return lambda text, cls: replies.pop(0)


GOOD = {"facts": [{"statement": "Profile data is written without a consent check.",
                   "fact_type": "partial", "evidence_ids": ["E001", "E999"], "confidence": 0.9}]}
EMPTY = {"facts": []}


def test_agent_drops_hallucinated_ids_and_retries():
    model = FakeChatModel(queue([OutputParserException("bad json")] + [GOOD] * 10))
    agent = ImplementationExtractionAgent(model)
    report = agent.run(APPS / "acmenotes_violating")
    assert model.calls == 6                  # 5 categories + 1 retry
    assert report.llm_retries == 1 and report.llm_failures == 0
    assert all("E999" not in f.evidence_ids for f in report.facts)
    assert report.facts[0].locations
    assert {f.category.value for f in report.facts} == {
        "transparency_notice", "consent", "purpose_limitation",
        "data_minimization", "storage_retention"}


def test_two_unusable_replies_are_counted_as_a_failure():
    err = OutputParserException("bad json")
    model = FakeChatModel(queue([err, err] + [EMPTY] * 4))
    agent = ImplementationExtractionAgent(model)
    report = agent.run(APPS / "acmenotes_baseline")
    assert report.llm_failures == 1 and report.llm_retries == 1
    assert agent.last_error and report.facts == []


def test_evidence_with_braces_does_not_break_the_prompt_template():
    seen = []

    def script(text, cls):
        seen.append(text)
        return EMPTY

    ImplementationExtractionAgent(FakeChatModel(script)).run(APPS / "acmenotes_violating")
    assert any("{" in t for t in seen)       # code snippets contain braces and still render

def test_scanner_detects_privacy_controls_in_static_javascript(tmp_path):
    js = tmp_path / "static" / "js" / "privacy.js"
    js.parent.mkdir(parents=True)
    js.write_text("""
const marketing = '<input type="checkbox" name="marketing_consent">';
const privacyLink = '<a href="/privacy">Privacy policy</a>';
const notice = "We collect your email to provide the service.";
const withdraw = "withdraw_consent";
""", encoding="utf-8")

    ev, files, cov = scan_project(tmp_path)

    assert files == 1
    assert cov["consent_ui"] >= 1
    assert cov["privacy_notice_link"] >= 1
    assert cov["purpose_text"] >= 1
    assert cov["form_input"] >= 1

def test_scanner_detects_django_model_and_view_data(tmp_path):
    accounts = tmp_path / "accounts"
    accounts.mkdir()

    (accounts / "models.py").write_text("""
from django.db import models

class SupportRequest(models.Model):
    email = models.EmailField()
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
""", encoding="utf-8")

    (accounts / "views.py").write_text("""
from django.http import JsonResponse

def profile(request):
    return JsonResponse({
        "user_id": 101,
        "email": "demo@example.test",
        "marketing_opt_in": False,
    })

def support(request):
    message = request.POST.get("message", "")
    return JsonResponse({"received": bool(message)})
""", encoding="utf-8")

    evidence, files, coverage = scan_project(tmp_path)

    assert files == 2
    assert coverage["model_field"] == 3
    assert coverage["view_response"] == 2
    assert coverage["request_input"] == 1

    fields = {
        item.detail["field"]
        for item in evidence
        if item.kind == "model_field"
    }
    assert fields == {"email", "message", "created_at"}

    request_fields = [
        item.detail["field"]
        for item in evidence
        if item.kind == "request_input"
    ]
    assert "message" in request_fields
