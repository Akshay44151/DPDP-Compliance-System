
# DPDP Compliance Evaluation Framework

## 1. Overview

This folder contains an independent evaluation framework for assessing the performance of Agent 3 (Alignment Agent) in the DPDP Compliance System.

The framework compares Agent 3's actual findings against manually prepared ground-truth answer keys for two seeded applications:

- `acmenotes_baseline`
- `acmenotes_violating`

It evaluates whether the agent correctly identifies policy-to-code alignment, undeclared data collection practices, and known seeded violations.

**Important:** This framework is separate from the existing deterministic compliance scoring engine and orchestrator. It does not modify or replace the compliance score.

## 2. Objectives

The evaluation framework is designed to:

1. Maintain ground-truth answer keys for both seeded applications.
2. Compare Agent 3's claim-level findings with expected alignment statuses.
3. Calculate alignment accuracy, precision, recall, and false-positive rate.
4. Evaluate expected undeclared data practices.
5. Check whether known seeded violations are detected.
6. Identify missing, ambiguous, and unexpected findings.

## 3. Folder Structure

```text
evaluation/
|
|-- ground_truth/
|   |-- acmenotes_baseline.json
|   |-- acmenotes_violating.json
|
|-- compare_findings.py
|-- compare_undeclared.py
|-- compare_seeded_violations.py
|-- metrics.py
|-- schema_mapping.py
|-- evaluate.py
|-- README.md
```

The corresponding automated tests are maintained separately under:

```text
tests/evaluation/
```

## 4. Ground-Truth Answer Keys

The ground-truth files contain manually curated expected results.

Each answer key includes:

- Application identifier
- Policy source
- Expected policy claim categories
- Expected alignment statuses
- Expected undeclared practices
- Known seeded violations
- Excluded policy content that should not be evaluated

The two answer keys represent different application behaviours:

| Application | Purpose |
|---|---|
| `acmenotes_baseline` | Represents the baseline application implementation. |
| `acmenotes_violating` | Represents an application containing deliberately seeded compliance violations. |

The answer keys are manually curated evaluation references and should be reviewed when the policy, application fixtures, or expected behaviour changes.

## 5. Evaluation Modules

### 5.1 Schema Mapping — `schema_mapping.py`

Maps expected answer-key labels to the existing project schemas:

- `AlignmentStatus`
- `PolicyCategory`

This avoids introducing a conflicting evaluation-specific status system.

### 5.2 Claim Comparison — `compare_findings.py`

Compares expected policy claims against Agent 3's actual claim-level findings.

Matching uses the existing project claim identifier generation function.

The comparison identifies:

- Correctly matched statuses
- Incorrect statuses
- Missing findings
- Duplicate or ambiguous findings

Undeclared-behaviour findings are handled separately.

### 5.3 Metrics — `metrics.py`

Calculates the following metrics:

| Metric | Meaning |
|---|---|
| Alignment accuracy | Proportion of evaluated in-scope claims with an exact status match. |
| Precision | Proportion of predicted violations that are expected violations. |
| Recall | Proportion of expected violations correctly identified. |
| False-positive rate | Proportion of expected non-violations incorrectly predicted as violations. |

For binary violation metrics, `violated` is treated as the positive class.

The current implementation treats other statuses as negative for this binary calculation. Missing and ambiguous findings are also reported separately and should be considered when interpreting the metrics.

### 5.4 Undeclared-Practice Comparison — `compare_undeclared.py`

Compares expected undeclared data practices against Agent 2 scanner evidence and Agent 3 findings.

It distinguishes between:

- `detected_undeclared`
- `reported_under_violation`
- `insufficient_evidence`
- `missed`
- `not_found_in_scan`

It also reports unexpected undeclared evidence.

**Detection rate clarification:** `detection_rate` measures explicit undeclared-practice detection only. Practices reported under an existing violated claim are tracked separately as `reported_under_violation` and are not counted as explicit undeclared detections. The denominator includes evaluable practices found by the scanner, including missed and insufficient-evidence cases.

Matching uses structured scanner evidence, including model-field names and personal-data logging tokens.

### 5.5 Seeded-Violation Comparison — `compare_seeded_violations.py`

Checks whether known seeded violations are identified.

It evaluates evidence-backed violations, such as unconditional marketing emails and external advertising transfers, as well as missing implementation behaviours such as retention and consent withdrawal.

It distinguishes between detected violations, missing scanner evidence, missed Agent findings, missing claim findings, and entries that cannot be independently evaluated.

**Reporting clarification:**

- `total` represents all seeded-violation entries, including those marked as `not_independently_evaluable`.
- `independently_evaluable` represents only entries that can be assessed independently.
- `counts["not_independently_evaluable"]` reports entries excluded from independent assessment.

The `total` field is a count of seeded-violation entries, not the detection-rate denominator. Entries marked `not_independently_evaluable` must not be interpreted as detected or missed violations.

### 5.6 Evaluation Entry Point — `evaluate.py`

Integrates the evaluation modules through the `evaluate_application()` function.

It returns:

- Claim-level comparisons
- Alignment metrics
- Undeclared-practice evaluation
- Seeded-violation evaluation

The implementation report is supplied separately so that scanner evidence can be used for undeclared-practice and seeded-violation evaluation.

## 6. Running the Tests

Activate the project's virtual environment, then run:

```powershell
python -m pytest tests\evaluation\ -v
```

To run a particular test module:

```powershell
python -m pytest tests\evaluation\test_evaluate.py -v
```

The tests cover claim matching, schema mapping, metric calculations, undeclared-practice detection, seeded-violation comparison, and integration between evaluation modules.

## 7. Using the Evaluation Framework

The main entry point is:

```python
from evaluation.evaluate import evaluate_application
```

The function accepts:

1. A ground-truth JSON file path.
2. Agent 3's alignment result.
3. An optional implementation report containing scanner evidence.

Example:

```python
from pathlib import Path
from evaluation.evaluate import evaluate_application

ground_truth_path = (
    Path("evaluation/ground_truth/acmenotes_violating.json")
)

result = evaluate_application(
    ground_truth_path=ground_truth_path,
    alignment_result=alignment_result,
    implementation_result=implementation_result,
)

print(result["metrics"])
print(result["undeclared_practice_evaluation"])
print(result["seeded_violation_evaluation"])
```

Here, `alignment_result` and `implementation_result` must be outputs from the corresponding project agents, or compatible objects containing their findings and evidence.

## 8. Scope and Limitations

- The framework evaluates Agent 3 findings against manually curated ground truth.
- It does not independently establish legal compliance.
- Ground-truth labels require review when application behaviour or policy content changes.
- Evaluation metrics depend on the correctness of the ground-truth answer keys.
- Unit tests using controlled sample inputs do not establish the actual performance of the AI agents.
- Real agent evaluation requires running the seeded applications through the existing agent pipeline and supplying their actual outputs.
- The framework does not change the existing compliance scoring engine or orchestrator.

## 9. Independence from Compliance Scoring

This evaluation framework is maintained separately from the deterministic compliance scoring engine.

Its purpose is to measure the quality of Agent 3's findings, not to calculate or replace the application's compliance score.

All evaluation results are intended for testing, validation, and improvement of the multi-agent compliance assessment system.
