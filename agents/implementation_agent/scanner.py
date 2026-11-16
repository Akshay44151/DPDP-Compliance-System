"""Deterministic static scanner for Django projects. Produces EvidenceItems."""
from __future__ import annotations

import ast
import re
from pathlib import Path

from agents.implementation_agent.schemas import CodeLocation, EvidenceItem

SKIP_DIRS = {"venv", ".venv", "env", "node_modules", "migrations", "__pycache__", ".git", "static"}

PERSONAL = {
    "email", "phone", "mobile", "aadhaar", "aadhar", "pan", "dob", "birth", "birthdate",
    "address", "pincode", "gender", "name", "firstname", "lastname", "passport",
    "location", "ip", "salary", "income", "religion", "caste", "health", "photo", "device",
}
CONSENT = {"consent", "optin", "opt", "agree", "agreed", "accept", "accepted", "withdraw", "withdrawn"}
RETENTION = {"retention", "expire", "expires", "expiry", "purge", "delete", "deletion",
             "anonymize", "anonymise", "cleanup", "erase", "erasure", "ttl", "archive"}
PURPOSE = {"purpose", "purposes", "usecase"}
WRITE_CALLS = {"create", "save", "bulk_create", "update_or_create", "get_or_create"}
SHARING_CALLS = {"requests.post", "requests.put", "httpx.post", "httpx.put"}
MAIL_CALLS = {"send_mail", "send_mass_mail", "EmailMessage", "EmailMultiAlternatives"}
LOG_METHODS = {"debug", "info", "warning", "error", "exception", "critical"}


def tokens(name: str) -> set[str]:
    return {t for t in re.split(r"[_\W]+", name.lower()) if t}


def dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{dotted(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        return dotted(node.func)
    return ""


class _Visitor(ast.NodeVisitor):
    def __init__(self, rel: str, source: str, add):
        self.rel, self.source, self.add = rel, source, add

    def snip(self, node: ast.AST) -> str:
        return (ast.get_source_segment(self.source, node) or "")[:300]

    def emit(self, kind, node, **detail):
        self.add(kind, self.rel, node.lineno, self.snip(node), detail)

    # ---- classes: models and forms ----
    def visit_ClassDef(self, node: ast.ClassDef):
        bases = {dotted(b).split(".")[-1] for b in node.bases}
        if tokens(node.name) & RETENTION:
            self.emit("retention_logic", node, name=node.name, role="class")
        if "Model" in bases:
            self._model_fields(node)
        elif bases & {"Form", "ModelForm"}:
            self._form(node)
        self.generic_visit(node)

    def _model_fields(self, cls: ast.ClassDef):
        for st in cls.body:
            if not (isinstance(st, ast.Assign) and isinstance(st.value, ast.Call)):
                continue
            ftype = dotted(st.value.func).split(".")[-1]
            if not ftype.endswith(("Field", "ForeignKey", "OneToOneField")):
                continue
            for t in st.targets:
                if not isinstance(t, ast.Name):
                    continue
                tk = tokens(t.id)
                kw = {k.arg: ast.unparse(k.value) for k in st.value.keywords if k.arg}
                d = dict(model=cls.name, field=t.id, field_type=ftype, options=kw)
                if tk & CONSENT:
                    self.emit("consent_logic", st, **d)
                elif tk & RETENTION:
                    self.emit("retention_field", st, **d)
                elif tk & PURPOSE:
                    self.emit("purpose_marker", st, **d)
                elif tk & PERSONAL or ftype == "EmailField":
                    self.emit("model_field", st, **d)

    def _form(self, cls: ast.ClassDef):
        for st in cls.body:
            if isinstance(st, ast.Assign) and isinstance(st.value, ast.Call):
                ftype = dotted(st.value.func).split(".")[-1]
                if ftype.endswith("Field"):
                    for t in st.targets:
                        if isinstance(t, ast.Name):
                            self.emit("form_field", st, form=cls.name, field=t.id, field_type=ftype)
            if isinstance(st, ast.ClassDef) and st.name == "Meta":
                for m in st.body:
                    if isinstance(m, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id in ("fields", "exclude") for t in m.targets
                    ):
                        self.emit("form_field", m, form=cls.name, meta=ast.unparse(m.value),
                                  option=m.targets[0].id)

    # ---- functions: writes, outbound email, consent/retention logic ----
    def _func(self, node):
        tk = tokens(node.name)
        if tk & RETENTION:
            self.emit("retention_logic", node, name=node.name, role="function")
        if tk & CONSENT:
            self.emit("consent_logic", node, name=node.name, role="function")
        names, writes, mails = set(), [], []
        for n in ast.walk(node):
            if isinstance(n, ast.Name):
                names |= tokens(n.id)
            elif isinstance(n, ast.Attribute):
                names |= tokens(n.attr)
            if isinstance(n, ast.Call):
                last = dotted(n.func).split(".")[-1]
                if isinstance(n.func, ast.Attribute) and n.func.attr in WRITE_CALLS:
                    writes.append(n.func.attr)
                if last in MAIL_CALLS:
                    mails.append(last)
        refs = dict(references_consent=bool(names & CONSENT), references_purpose=bool(names & PURPOSE))
        if writes:
            self.emit("data_write", node, function=node.name, calls=sorted(set(writes)), **refs)
        if mails:
            self.emit("outbound_email", node, function=node.name, calls=sorted(set(mails)), **refs)
        self.generic_visit(node)

    visit_FunctionDef = _func
    visit_AsyncFunctionDef = _func

    # ---- calls: third-party sharing, personal data in logs ----
    def visit_Call(self, node: ast.Call):
        name = dotted(node.func)
        if name in SHARING_CALLS:
            self.emit("data_sharing", node, call=name)
        if isinstance(node.func, ast.Attribute) and node.func.attr in LOG_METHODS \
                and "log" in dotted(node.func.value).lower():
            hit = tokens(" ".join(ast.unparse(a) for a in node.args)) & PERSONAL
            if hit:
                self.emit("personal_data_logged", node, personal_tokens=sorted(hit))
        self.generic_visit(node)


def _scan_settings(rel: str, source: str, tree: ast.Module, add):
    for st in tree.body:
        if isinstance(st, ast.Assign):
            for t in st.targets:
                if isinstance(t, ast.Name) and t.id.isupper():
                    if tokens(t.id) & (RETENTION | CONSENT | {"session", "logging", "cookie"}):
                        add("setting", rel, st.lineno,
                            (ast.get_source_segment(source, st) or "")[:300], {"name": t.id})


def _scan_template(rel: str, text: str, add):
    for i, line in enumerate(text.splitlines(), 1):
        low = line.lower()
        if re.search(r"<input[^>]*type=[\"']checkbox[\"']", low):
            kind = "consent_ui" if tokens(re.sub(r"[^a-z_ ]", " ", low)) & CONSENT else "form_input"
            add(kind, rel, i, line.strip()[:300], {"input": "checkbox"})
        elif m := re.search(r"<input[^>]*name=[\"']([\w\-]+)[\"']", low):
            add("form_input", rel, i, line.strip()[:300], {"name": m.group(1)})
        if re.search(r"href=[\"'][^\"']*privacy[^\"']*[\"']", low):
            add("privacy_notice_link", rel, i, line.strip()[:300], {})
        if "purpose" in low or "we use your" in low or "we collect" in low:
            add("purpose_text", rel, i, line.strip()[:300], {})


def scan_project(root: str | Path):
    """Return (evidence, files_scanned, coverage)."""
    root = Path(root)
    evidence: list[EvidenceItem] = []

    def add(kind, rel, line, snippet, detail):
        evidence.append(EvidenceItem(id=f"E{len(evidence)+1:03d}", kind=kind,
                                     location=CodeLocation(file=rel, line=line),
                                     snippet=snippet, detail=detail))

    files = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(p in SKIP_DIRS for p in path.relative_to(root).parts):
            continue
        rel = path.relative_to(root).as_posix()
        if path.suffix == ".py":
            source = path.read_text(encoding="utf-8", errors="ignore")
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            files += 1
            if tokens(path.stem) & RETENTION:  # e.g. management/commands/purge_inactive_accounts.py
                add("retention_logic", rel, 1, "", {"name": path.stem, "role": "file"})
            _Visitor(rel, source, add).visit(tree)
            if path.name.startswith("settings"):
                _scan_settings(rel, source, tree, add)
        elif path.suffix in (".html", ".htm"):
            files += 1
            _scan_template(rel, path.read_text(encoding="utf-8", errors="ignore"), add)

    coverage: dict[str, int] = {}
    for e in evidence:
        coverage[e.kind] = coverage.get(e.kind, 0) + 1
    return evidence, files, coverage
