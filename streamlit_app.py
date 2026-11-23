import json

from dashboard.pdf_report import make_pdf
from contextlib import redirect_stdout
import hmac
from pathlib import Path, PurePosixPath
import io
import os
import stat
import tempfile
import zipfile

import streamlit as st

from scripts.run_assessment import run_full_real
from agents.legal_rag.ingest import CHROMA_DIR, ingest_legal_documents


# --------------------------------------------------
# Upload limits
# --------------------------------------------------

MAX_ZIP_SIZE = 100 * 1024 * 1024
MAX_POLICY_SIZE = 2 * 1024 * 1024
MAX_EXTRACTED_SIZE = 300 * 1024 * 1024
MAX_ZIP_FILES = 10000


# --------------------------------------------------
# Password protection
# --------------------------------------------------

def check_password():
    """Require a password before allowing access to the app."""

    if st.session_state.get("authenticated", False):
        return

    try:
        expected_password = str(st.secrets["APP_PASSWORD"])
    except (KeyError, FileNotFoundError):
        st.error("App password is not configured.")
        st.stop()

    if not expected_password:
        st.error("App password is empty. Please contact the administrator.")
        st.stop()

    def password_entered():
        entered_password = st.session_state.get("password_input", "")

        if hmac.compare_digest(entered_password, expected_password):
            st.session_state["authenticated"] = True
            st.session_state.pop("password_error", None)
        else:
            st.session_state["password_error"] = True

    st.text_input(
        "Enter app password",
        type="password",
        key="password_input",
        on_change=password_entered,
    )

    if st.session_state.get("password_error", False):
        st.error("Incorrect password. Please try again.")

    st.info("Enter your access password to use the DPDP assessment system.")
    st.stop()


# --------------------------------------------------
# Safe ZIP extraction
# --------------------------------------------------

def extract_project_zip(zip_bytes, destination):
    """Validate and safely extract an uploaded project ZIP."""

    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        entries = archive.infolist()

        if len(entries) > MAX_ZIP_FILES:
            raise ValueError("ZIP contains too many files.")

        total_size = sum(entry.file_size for entry in entries)

        if total_size > MAX_EXTRACTED_SIZE:
            raise ValueError("Extracted project exceeds the 300 MB limit.")

        for entry in entries:
            normalized_name = entry.filename.replace("\\", "/")
            member = PurePosixPath(normalized_name)

            if (
                member.is_absolute()
                or ".." in member.parts
                or not member.parts
                or any(":" in part for part in member.parts)

            ):
                raise ValueError("ZIP contains an unsafe file path.")

            if stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError("ZIP contains a symbolic link.")

            target = (destination / Path(*member.parts)).resolve()

            try:
                target.relative_to(destination)
            except ValueError:
                raise ValueError(
                    "ZIP contains a path outside its destination."
                )

            # Reject special file types other than regular files/directories.
            unix_mode = entry.external_attr >> 16
            file_type = stat.S_IFMT(unix_mode)

            if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise ValueError("ZIP contains an unsupported file type.")

        archive.extractall(destination)


# --------------------------------------------------
# Detect project root
# --------------------------------------------------

def find_project_root(extraction_folder):
    """Handle ZIP files that contain one enclosing project directory."""

    entries = list(extraction_folder.iterdir())

    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]

    return extraction_folder


# --------------------------------------------------
# Legal database initialization
# --------------------------------------------------

@st.cache_resource(show_spinner=False)
def initialize_legal_database():
    """Build the legal database once when it is missing."""

    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        ingest_legal_documents()

    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        raise RuntimeError("Legal database initialization failed.")

    return True


# --------------------------------------------------
# Streamlit page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="DPDP Compliance Assessment",
    page_icon="🔐",
    layout="wide",
)

check_password()


# --------------------------------------------------
# Load API configuration securely
# --------------------------------------------------

def configure_llm_environment():
    """Load Groq settings securely from Streamlit secrets."""

    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
        llm_model = st.secrets["LLM_MODEL"]
    except (KeyError, FileNotFoundError):
        st.error("Groq API key or model is missing from app secrets.")
        st.stop()

    if not groq_api_key or not llm_model:
        st.error("Groq API key or model is empty.")
        st.stop()

    os.environ["GROQ_API_KEY"] = str(groq_api_key)
    os.environ["LLM_MODEL"] = str(llm_model)
    os.environ["LLM_PROVIDER"] = "groq"


configure_llm_environment()


# --------------------------------------------------
# Initialize legal reference database
# --------------------------------------------------

try:
    with st.spinner(
        "Preparing DPDP legal reference database. "
        "First startup may take a few minutes..."
    ):
        initialize_legal_database()

except Exception as error:
    print(
        f"Legal database initialization failed: "
        f"{type(error).__name__}"
    )

    st.error(
        "The legal reference database could not be initialized. "
        "Please contact the app administrator."
    )
    st.stop()


# --------------------------------------------------
# Main page
# --------------------------------------------------

st.title("DPDP Compliance Assessment System")

st.write(
    "Upload a Django project and its privacy policy "
    "to run an automated DPDP compliance assessment."
)

st.warning(
    "Full assessment uses real Groq API calls. "
    "It may incur API charges. Upload only code you are "
    "authorized to analyze."
)


# --------------------------------------------------
# Project upload
# --------------------------------------------------

st.subheader("1. Upload your project")

project_zip = st.file_uploader(
    "Django project ZIP",
    type=["zip"],
    help="Upload your project as a ZIP file. Maximum size: 100 MB.",
)


# --------------------------------------------------
# Privacy policy upload
# --------------------------------------------------

st.subheader("2. Upload your privacy policy")

policy_file = st.file_uploader(
    "Privacy policy TXT file",
    type=["txt"],
    help="Upload your privacy policy as a plain-text file. Maximum size: 2 MB.",
)


# --------------------------------------------------
# Assessment confirmation
# --------------------------------------------------

st.divider()

st.subheader("3. Run the assessment")

with st.form("assessment_form"):

    confirmed = st.checkbox(
        "I understand that this assessment makes real Groq API calls "
        "and may incur charges."
    )

    submitted = st.form_submit_button(
        "Run Full Assessment",
        type="primary",
    )


# --------------------------------------------------
# Execute assessment
# --------------------------------------------------


if submitted:

    # Clear previous results before processing a new submission.
    for key in (
        "assessment_finished",
        "assessment_log",
        "assessment_exit_code",
        "audit_bytes",
    ):
        st.session_state.pop(key, None)

    if project_zip is None:
        st.error("Please upload a project ZIP file.")

    elif policy_file is None:
        st.error("Please upload a privacy policy TXT file.")

    elif not confirmed:
        st.error("Please confirm the API usage warning.")

    elif len(project_zip.getvalue()) > MAX_ZIP_SIZE:
        st.error("Project ZIP exceeds the 100 MB upload limit.")

    elif len(policy_file.getvalue()) > MAX_POLICY_SIZE:
        st.error("Privacy policy exceeds the 2 MB upload limit.")

    else:

        try:
            log_buffer = io.StringIO()

            with tempfile.TemporaryDirectory(
                prefix="dpdp_streamlit_"
            ) as temporary_folder:

                work_folder = Path(temporary_folder)

                extraction_folder = work_folder / "project"
                output_folder = work_folder / "audit_output"

                output_folder.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                with st.spinner("Preparing project files..."):

                    extract_project_zip(
                        project_zip.getvalue(),
                        extraction_folder,
                    )

                    project_folder = find_project_root(
                        extraction_folder
                    )

                    policy_path = work_folder / "privacy_policy.txt"

                    policy_path.write_bytes(
                        policy_file.getvalue()
                    )

                st.info(
                    "Project files prepared. "
                    "Starting the full assessment."
                )

                with st.spinner(
                    "Running Agents 1–4, evidence gates, scoring and reporting..."
                ):

                    with redirect_stdout(log_buffer):

                        exit_code = run_full_real(
                            project=project_folder,
                            policy=policy_path,
                            output_dir=output_folder,
                        )

                # Read audit JSON before temporary files are deleted.

                audit_bytes = None

                audit_files = list(
                    output_folder.rglob("*.json")
                )

                if audit_files:

                    latest_audit = max(
                        audit_files,
                        key=lambda path: path.stat().st_mtime,
                    )

                    audit_bytes = latest_audit.read_bytes()

            # TemporaryDirectory automatically removes uploaded
            # project files, policy and audit output here.

            st.session_state["assessment_log"] = (
                log_buffer.getvalue()
            )

            st.session_state["assessment_exit_code"] = exit_code

            st.session_state["audit_bytes"] = audit_bytes

            st.session_state["assessment_finished"] = True

        except zipfile.BadZipFile:

            st.session_state["assessment_finished"] = False

            st.error(
                "The uploaded file is not a valid ZIP archive."
            )

        except Exception as error:

            st.session_state["assessment_finished"] = False

            st.session_state["assessment_log"] = (
                log_buffer.getvalue()
            )

            print(
                f"Assessment failed: {type(error).__name__}"
            )

            st.error(
                "The assessment could not be completed. "
                f"Error type: {type(error).__name__}. "
                "Please contact the app administrator."
            )



# --------------------------------------------------
# Display integrated assessment results and reports
# --------------------------------------------------

if st.session_state.get("assessment_finished", False):

    st.divider()
    st.header("Assessment Results")

    if st.session_state["assessment_exit_code"] == 0:
        st.success("Assessment completed successfully.")
    else:
        st.warning(
            "Assessment is incomplete or encountered an error. "
            "Review the execution log and available audit record."
        )

    audit_bytes = st.session_state.get("audit_bytes")

    if audit_bytes:

        try:
            audit_data = json.loads(audit_bytes.decode("utf-8-sig"))

            if not isinstance(audit_data, dict):
                raise ValueError("Audit record must be a JSON object.")

        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            st.error(f"Could not read the generated audit record: {error}")
            audit_data = None

        if audit_data is not None:

            summary = audit_data.get("summary")
            summary = summary if isinstance(summary, dict) else {}

            score = audit_data.get("compliance_score")
            score = score if isinstance(score, dict) else {}

            compliance = audit_data.get("compliance_report")
            compliance = compliance if isinstance(compliance, dict) else {}

            gated = audit_data.get("legal_analysis_gated")
            gated = gated if isinstance(gated, dict) else {}

            findings = gated.get("findings", [])
            findings = findings if isinstance(findings, list) else []

            # Assessment overview
            st.subheader("1. Assessment Overview")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Total findings",
                    score.get("total_findings", len(findings)),
                )

            with col2:
                st.metric(
                    "Assessed findings",
                    compliance.get("assessed_findings", "Not available"),
                )

            with col3:
                st.metric(
                    "Assessment coverage",
                    f"{score.get('assessment_coverage_percent', 'Not available')}%",
                )

            st.write(
                "**Assessment ID:**",
                audit_data.get("assessment_id", "Not available"),
            )

            if score.get("score") is None:
                st.warning(
                    "Compliance score withheld: "
                    + str(
                        score.get(
                            "score_withheld_reason",
                            "No legally assessable outcome is available.",
                        )
                    )
                )
            else:
                st.metric("Compliance Score", score["score"])

            # Final legally gated findings
            st.divider()
            st.subheader("2. Final Legal Findings")

            st.caption(
                "These findings reflect the final evidence-gated legal analysis."
            )

            if not findings:
                st.info("No final legal findings are available.")
            else:
                table_rows = []

                for index, finding in enumerate(findings, start=1):
                    if not isinstance(finding, dict):
                        continue

                    table_rows.append({
                        "Finding": finding.get(
                            "alignment_finding_id",
                            f"Finding {index}",
                        ),
                        "Status": finding.get("status", "Not available"),
                        "Confidence": finding.get("confidence", "Not available"),
                    })

                if table_rows:
                    st.dataframe(
                        table_rows,
                        use_container_width=True,
                        hide_index=True,
                    )

                for index, finding in enumerate(findings, start=1):
                    if not isinstance(finding, dict):
                        continue

                    finding_id = finding.get(
                        "alignment_finding_id",
                        f"Finding {index}",
                    )
                    status = finding.get("status", "Not available")

                    with st.expander(f"{finding_id} — {status}"):

                        for field, label in [
                            ("legal_reasoning", "Legal reasoning"),
                            ("applicable_provisions", "Applicable provisions"),
                            ("evidence_fact_ids", "Evidence references"),
                            ("citations", "Legal citations"),
                            ("confidence", "Confidence"),
                            ("limitations", "Limitations"),
                        ]:
                            if field in finding:
                                st.markdown(f"**{label}**")
                                st.write(finding[field])

            # Evidence and implementation
            st.divider()
            st.subheader("3. Evidence and Implementation")

            for key, label in [
                ("implementation_report", "Implementation analysis"),
                ("evidence_gate", "Evidence gate results"),
                ("legal_gate_application", "Legal gate decisions"),
            ]:
                section = audit_data.get(key)

                with st.expander(label):
                    if isinstance(section, dict) and section:
                        for field, value in section.items():
                            st.markdown(
                                f"**{str(field).replace('_', ' ').title()}**"
                            )
                            st.write(value)
                    elif section:
                        st.write(section)
                    else:
                        st.caption("No information available.")

            # Recommendations
            st.divider()
            st.subheader("4. Recommendations")

            recommendations = compliance.get("recommendations", [])

            if not recommendations:
                recommendations = audit_data.get("recommendations", [])

            if isinstance(recommendations, list) and recommendations:
                for recommendation in recommendations:
                    if isinstance(recommendation, dict):
                        title = (
                            recommendation.get("finding_id")
                            or recommendation.get("title")
                            or "Recommendation"
                        )
                        with st.expander(str(title)):
                            for key, value in recommendation.items():
                                st.markdown(
                                    f"**{str(key).replace('_', ' ').title()}**"
                                )
                                st.write(value)
                    else:
                        st.write(recommendation)
            else:
                st.info("No recommendations were recorded.")

            # Downloads
            st.divider()
            st.subheader("5. Download Assessment Reports")

            col1, col2 = st.columns(2)

            with col1:
                st.download_button(
                    "Download Audit JSON",
                    data=audit_bytes,
                    file_name="dpdp_audit_record.json",
                    mime="application/json",
                    use_container_width=True,
                    key="download_audit_json",
                )

            with col2:
                try:
                    pdf_bytes = make_pdf(audit_data)

                    st.download_button(
                        "Download PDF Report",
                        data=pdf_bytes,
                        file_name="dpdp_compliance_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        key="download_audit_pdf",
                    )

                except Exception as error:
                    st.error(
                        "PDF generation failed. "
                        f"Error type: {type(error).__name__}"
                    )

    else:
        st.info("No audit JSON file was found in the output folder.")

    # Execution log
    st.divider()

    with st.expander("View Assessment Execution Log"):
        st.text_area(
            "Execution log",
            value=st.session_state.get("assessment_log", ""),
            height=350,
            disabled=True,
        )

    st.caption(
        "This is an automated informational assessment, not legal advice "
        "or a legal certification. Findings should be reviewed by qualified personnel."
    )
