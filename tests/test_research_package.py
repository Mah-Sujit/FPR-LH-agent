import json

from foresight_agent.research.artifacts import environment_manifest, sha256, write_checksums
from foresight_agent.research.pipeline import run_pipeline


def test_checksum_and_environment_manifest(tmp_path):
    source = tmp_path / "sample.txt"
    source.write_text("foresight", encoding="utf-8")
    assert len(sha256(source)) == 64
    manifest = environment_manifest()
    assert "python" in manifest and "packages" in manifest
    output = tmp_path / "checksums.json"
    write_checksums(tmp_path, output)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["files"]["sample.txt"] == sha256(source)


def test_end_to_end_research_package(tmp_path):
    output = run_pipeline(
        dataset_path="data/collection/v08/provisional_labeled.jsonl",
        output_root=tmp_path,
    )
    expected = {"MODEL_CARD.md", "DATASET_CARD.md", "audit.json", "checksums.json",
                "config_snapshot.json", "environment.json", "metrics.svg", "run_manifest.json"}
    assert expected <= {path.name for path in output.iterdir()}
    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    assert audit["passed"] is True
    checksums = json.loads((output / "checksums.json").read_text(encoding="utf-8"))
    assert "metrics.svg" in checksums["files"]
