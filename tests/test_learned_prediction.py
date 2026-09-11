import json

from foresight_agent.agent.state import StepRecord
from foresight_agent.config import ExperimentConfig
from foresight_agent.evaluation.training import _split, train_models
from foresight_agent.prediction.features import FEATURE_NAMES, feature_vector
from foresight_agent.prediction.learned import LearnedRiskScorer


def test_feature_schema_is_stable():
    record = StepRecord(1, "goal", "action", "observation", True, confidence=0.8)
    record.signals = {"low_confidence": 0.2}
    vector = feature_vector(record)
    assert len(vector) == len(FEATURE_NAMES)
    assert all(isinstance(value, float) for value in vector)


def test_training_saves_model_and_leakage_free_splits(tmp_path):
    config = ExperimentConfig(seed=11, noisy_runs_per_task=3)
    paths = train_models(config, tmp_path, runs=5)
    assert all(path.exists() for path in paths.values())
    manifest = json.loads(paths["splits"].read_text(encoding="utf-8"))
    train, validation, test = map(set, (
        manifest["train_groups"], manifest["validation_groups"], manifest["test_groups"]))
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)
    scorer = LearnedRiskScorer(paths["model"])
    assert 0.0 <= scorer.score_record(None, StepRecord(
        1, "goal", "action", "observation", True, confidence=0.9)) <= 1.0


def test_group_split_places_both_classes_in_every_partition():
    rows, labels, groups = [], [], []
    for group in range(20):
        for label in (0, 1) if group < 6 else (0, 0):
            rows.append([float(group)]); labels.append(label); groups.append(f"task-{group}")
    for partition in _split(rows, labels, groups, seed=42):
        assert set(labels[index] for index in partition) == {0, 1}
