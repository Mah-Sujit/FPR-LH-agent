import json

from foresight_agent.data.external import load_trajectories, write_normalized_jsonl
from foresight_agent.research.release import evaluate_release_gate, write_release_draft


def test_release_gate_rejects_bootstrap_and_accepts_human_evidence(tmp_path):
    package = tmp_path / "package"
    model = package / "model"; model.mkdir(parents=True)
    (package / "audit.json").write_text('{"passed": true}', encoding="utf-8")
    (package / "run_manifest.json").write_text(
        '{"run_id":"test","dataset_sha256":"abc"}', encoding="utf-8")
    (model / "external_metadata.json").write_text(
        '{"selected_model":"logistic_regression","threshold":0.5}', encoding="utf-8")
    (model / "external_metrics.csv").write_text(
        "model,precision,recall,f1,roc_auc,pr_auc,brier,calibration_error\n"
        "logistic_regression,0.8,0.7,0.75,0.85,0.82,0.15,0.05\n", encoding="utf-8")

    bootstrap = "data/collection/v08/provisional_labeled.jsonl"
    assert evaluate_release_gate(package, bootstrap)["ready_for_1_0"] is False
    items = load_trajectories(bootstrap, require_labels=True)
    for item in items:
        item.annotator_id = "agreement:reviewer-a+reviewer-b"
        item.source = "human-reviewed-collection"
    human = tmp_path / "human.jsonl"; write_normalized_jsonl(items, human)
    agreement = tmp_path / "agreement.json"
    agreement.write_text(json.dumps({"rows": len(items), "binary_cohens_kappa": 0.8,
                                     "annotator_a": ["reviewer-a"], "annotator_b": ["reviewer-b"]}),
                         encoding="utf-8")
    assert evaluate_release_gate(package, human, agreement)["ready_for_1_0"] is True
    report = write_release_draft(package, human, tmp_path / "report.md")
    assert "HUMAN-LABELED CANDIDATE" in report.read_text(encoding="utf-8")
