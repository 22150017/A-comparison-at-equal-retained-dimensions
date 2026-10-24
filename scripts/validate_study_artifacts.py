
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.reproducibility import canonical_hash, file_sha256
from scripts.validation_math import confusion_errors

PRUNE_RECEIPT_NAME = "rejected_diagnostic_prune_receipt.json"


def validate_rejected_prune_receipt(manifest_path: Path, payload: dict) -> list[str]:
    errors: list[str] = []
    receipt_path = manifest_path.parent / PRUNE_RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        not receipt.get("complete")
        or receipt.get("status") != "complete"
        or not receipt.get("archive_signature")
    ):
        return [f"incomplete rejected-diagnostic prune receipt: {receipt_path}"]
    if not receipt.get("claim_boundary") or receipt.get("schema_version") != 1:
        errors.append(f"prune receipt lacks schema/claim boundary: {receipt_path}")
    signature = str(payload.get("signature", ""))
    matching = [
        item for item in receipt.get("targets", [])
        if str(item.get("dataset_signature")) == signature
    ]
    if len(matching) != 1:
        return [f"prune receipt does not uniquely cover manifest {signature}: {receipt_path}"]
    target = matching[0]
    current_hash = file_sha256(manifest_path)
    if current_hash != target.get("manifest_sha256"):
        errors.append(f"pruned manifest hash changed: {manifest_path}")
    archived_manifest = Path(str(target.get("archived_manifest", "")))
    if (
        not archived_manifest.is_file()
        or file_sha256(archived_manifest) != target.get("archived_manifest_sha256")
        or target.get("archived_manifest_sha256") != target.get("manifest_sha256")
    ):
        errors.append(f"archived manifest copy is missing or changed: {archived_manifest}")
    records = payload.get("records", [])
    if len(records) != int(target.get("record_count", -1)):
        errors.append(f"pruned manifest record count changed: {manifest_path}")
    existing_images = [
        Path(str(record.get("image_path", "")))
        for record in records
        if Path(str(record.get("image_path", ""))).is_file()
    ]
    if existing_images or int(target.get("remaining_dataset_images", -1)) != 0:
        errors.append(f"pruned dataset unexpectedly retains images: {manifest_path.parent}")
    expected_deleted = sum(int(item.get("record_count", 0)) for item in receipt.get("targets", []))
    if int(receipt.get("deleted_dataset_images", -1)) != expected_deleted:
        errors.append(f"prune receipt deleted-image count mismatch: {receipt_path}")
    for evidence in target.get("preserved_evidence", []):
        path = Path(str(evidence.get("path", "")))
        if (
            not path.is_file()
            or path.stat().st_size != int(evidence.get("size", -1))
            or file_sha256(path) != evidence.get("sha256")
        ):
            errors.append(f"preserved rejection evidence is missing or changed: {path}")
    config_path = Path(str(receipt.get("config_path", "")))
    if not config_path.is_file() or file_sha256(config_path) != receipt.get("config_sha256"):
        errors.append(f"prune archive config is missing or changed: {config_path}")
    return errors


def validate_manifest(path: Path) -> list[str]:
    errors: list[str] = []
    payload = json.loads(path.read_text(encoding="utf-8"))
    prune_receipt = path.parent / PRUNE_RECEIPT_NAME
    intentionally_pruned = prune_receipt.is_file()
    if intentionally_pruned:
        errors.extend(validate_rejected_prune_receipt(path, payload))
    if not payload.get("complete"):
        errors.append(f"incomplete manifest: {path}")
    split_metadata = payload.get("condition", {}).get("split", {})
    deliberately_leaky = split_metadata.get("mode") == "deliberately_leaky_window_control"
    if deliberately_leaky and split_metadata.get("paper_claim_authorized") is not False:
        errors.append(f"leaky control is not explicitly blocked from paper claims: {path}")
    groups = {name: set(values) for name, values in payload.get("groups", {}).items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = groups.get(left, set()) & groups.get(right, set())
        if overlap and not deliberately_leaky:
            errors.append(f"group leakage {left}/{right} in {path}: {sorted(overlap)[:5]}")
    records = payload.get("records", [])
    split_unit = split_metadata.get("unit")
    actual_split_groups: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    source_splits: dict[str, set[str]] = {}
    for record in records:
        split = str(record.get("split", ""))
        if split not in actual_split_groups:
            continue
        raw_source = record.get("source_file")
        
        
        source = str(raw_source) if raw_source else ""
        split_group = str(record.get("split_group_id", record.get("group_id", "")))
        actual_split_groups[split].add(split_group)
        if source:
            source_splits.setdefault(source, set()).add(split)
        if split_unit == "source_file":
            if "split_group_id" not in record:
                errors.append(f"missing split_group_id for source-file split in {path}")
                break
            if not source or split_group != source:
                errors.append(
                    f"source-file split_group_id mismatch in {path}: "
                    f"recorded={split_group}, source={source}"
                )
                break
    for split in ("train", "val", "test"):
        if groups.get(split, set()) != actual_split_groups[split]:
            errors.append(
                f"manifest group summary mismatch for {split} in {path}: "
                f"recorded={len(groups.get(split, set()))}, actual={len(actual_split_groups[split])}"
            )
    leaking_sources = {
        source: sorted(splits) for source, splits in source_splits.items() if len(splits) > 1
    }
    if leaking_sources and not deliberately_leaky:
        first_source = sorted(leaking_sources)[0]
        errors.append(
            f"source recording leakage in {path}: {first_source} -> {leaking_sources[first_source]}"
        )
    if deliberately_leaky and not leaking_sources:
        errors.append(f"declared leaky control does not actually mix source recordings: {path}")
    valid_statuses = {"generated", "reused_valid_image"}
    if any(record.get("status") not in valid_statuses for record in records):
        errors.append(f"failed/non-recoverable records in {path}")
    if not intentionally_pruned:
        for record in records:
            image_path = Path(record.get("image_path", ""))
            if not image_path.is_file():
                errors.append(f"missing image: {image_path}")
                break
    explicit = payload.get("condition", {}).get("explicit_holdout")
    if explicit:
        expected_test = set(explicit.get("test_groups", []))
        expected_val = set(explicit.get("val_groups", []))
        actual_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
        group_labels: dict[str, set[str]] = {}
        for record in records:
            split = str(record.get("split", ""))
            group = str(record.get("group_id", ""))
            label = str(record.get("label", ""))
            if split in actual_by_split:
                actual_by_split[split].add(group)
            group_labels.setdefault(group, set()).add(label)
        all_groups = set(group_labels)
        expected_train = all_groups - expected_test - expected_val
        for split, expected in (
            ("train", expected_train),
            ("val", expected_val),
            ("test", expected_test),
        ):
            if actual_by_split[split] != expected:
                errors.append(
                    f"explicit holdout mismatch for {split} in {path}: "
                    f"expected={sorted(expected)}, actual={sorted(actual_by_split[split])}"
                )
        mixed = {group: sorted(labels) for group, labels in group_labels.items() if len(labels) != 1}
        if mixed:
            errors.append(f"groups contain multiple labels in {path}: {mixed}")
        all_labels = {next(iter(labels)) for labels in group_labels.values() if len(labels) == 1}
        for split in ("train", "test"):
            split_labels = {
                next(iter(group_labels[group]))
                for group in actual_by_split[split]
                if len(group_labels.get(group, set())) == 1
            }
            if split_labels != all_labels:
                errors.append(
                    f"explicit {split} split does not cover all classes in {path}: "
                    f"missing={sorted(all_labels - split_labels)}"
                )
        recorded_val_classes = set(explicit.get("val_classes", []))
        actual_val_classes = {
            next(iter(group_labels[group]))
            for group in actual_by_split["val"]
            if len(group_labels.get(group, set())) == 1
        }
        if recorded_val_classes != actual_val_classes:
            errors.append(
                f"explicit val_classes mismatch in {path}: "
                f"recorded={sorted(recorded_val_classes)}, actual={sorted(actual_val_classes)}"
            )
    phase2 = payload.get("condition", {}).get("phase2", {})
    cross = phase2.get("cross_condition")
    if cross:
        train_conditions = set(cross.get("train_conditions", []))
        test_condition = cross.get("test_condition")
        for record in records:
            parts = Path(record.get("source_file", "")).stem.split("_")
            operating_condition = "_".join(parts[:3]) if len(parts) >= 3 else "unknown"
            split = record.get("split")
            if split in {"train", "val"} and operating_condition not in train_conditions:
                errors.append(f"cross-condition source-domain violation in {path}: {record}")
                break
            if split == "test" and operating_condition != test_condition:
                errors.append(f"cross-condition target-domain violation in {path}: {record}")
                break
    if phase2.get("compression_baseline") == "pca":
        metadata_path = path.parent / "pca_training_only.json"
        if not metadata_path.is_file():
            errors.append(f"missing training-only PCA metadata: {metadata_path}")
        else:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("fit_scope") != "training_only":
                errors.append(f"PCA fit scope is not training_only: {metadata_path}")
    diagnostic_config = phase2.get("diagnostic_prior")
    if diagnostic_config:
        if payload.get("condition", {}).get("method") != "dp_ilsomp":
            errors.append(f"diagnostic prior is attached to a non-DP method: {path}")
        metadata_path = path.parent / "diagnostic_prior.json"
        array_path = path.parent / "diagnostic_prior.npz"
        if not metadata_path.is_file() or not array_path.is_file():
            errors.append(f"missing diagnostic-prior artifact beside {path}")
        else:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            recorded = payload.get("diagnostic_prior", {})
            if not metadata.get("complete") or metadata.get("config", {}).get("fit_scope") != "training_only":
                errors.append(f"diagnostic prior is incomplete or not training-only: {metadata_path}")
            if recorded.get("prior_signature") != metadata.get("prior_signature"):
                errors.append(f"manifest diagnostic-prior signature mismatch: {path}")
            actual_hash = file_sha256(array_path)
            if metadata.get("npz_sha256") != actual_hash or recorded.get("npz_sha256") != actual_hash:
                errors.append(f"diagnostic-prior content hash mismatch: {array_path}")
            try:
                with np.load(array_path, allow_pickle=False) as prior:
                    selected_ids = [str(value) for value in prior["selected_sample_ids"].tolist()]
                    weights = np.asarray(prior["atom_weights"], dtype=np.float64)
                    class_counts = np.asarray(prior["class_counts"], dtype=np.int64)
                train_ids = {str(record.get("sample_id")) for record in records if record.get("split") == "train"}
                if not set(selected_ids).issubset(train_ids):
                    errors.append(f"diagnostic prior contains non-training samples: {array_path}")
                if diagnostic_config.get("fit_sample_count") != len(selected_ids):
                    errors.append(f"diagnostic-prior fit sample count mismatch: {path}")
                if diagnostic_config.get("fit_sample_ids_hash") != canonical_hash(selected_ids, length=32):
                    errors.append(f"diagnostic-prior fit sample hash mismatch: {path}")
                expected_length = int(payload.get("condition", {}).get("signal", {}).get("segment_length", 0))
                if weights.ndim != 1 or (expected_length and len(weights) != expected_length):
                    errors.append(f"diagnostic-prior atom length mismatch: {array_path}")
                if np.any(class_counts < 2) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
                    errors.append(f"invalid diagnostic-prior values: {array_path}")
            except (KeyError, ValueError, OSError) as exc:
                errors.append(f"cannot validate diagnostic-prior payload {array_path}: {exc}")
    return errors


def validate_metrics(path: Path) -> list[str]:
    errors: list[str] = []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("evaluation_scope") == "validation-only":
        if payload.get("independent_test_opened") is not False:
            errors.append(f"validation-only run opened the independent test: {path}")
        if payload.get("dataloaded_splits") != ["train", "val"]:
            errors.append(f"validation-only run constructed unexpected loaders: {path}")
        forbidden = {"metrics", "paper_metrics", "test_metrics"} & set(payload)
        if forbidden:
            errors.append(f"validation-only run contains forbidden test fields {sorted(forbidden)}: {path}")
        metrics = payload.get("validation_metrics")
        if metrics != payload.get("val_metrics") or not isinstance(metrics, dict):
            errors.append(f"validation-only metrics/val_metrics mismatch: {path}")
            return errors
        errors.extend(confusion_errors(metrics, payload.get("class_names")))
        matrix = metrics.get("confusion_matrix")
        sample_count = metrics.get("sample_count")
        if not isinstance(matrix, list) or not matrix:
            errors.append(f"missing validation confusion matrix: {path}")
        elif sum(sum(int(value) for value in row) for row in matrix) != sample_count:
            errors.append(f"validation confusion matrix total != sample count: {path}")
        if not payload.get("run_signature"):
            errors.append(f"missing validation run signature: {path}")
        return errors
    if payload.get("metrics") != payload.get("test_metrics"):
        errors.append(f"metrics is not the independent test result: {path}")
    if payload.get("paper_metrics") != payload.get("test_metrics"):
        errors.append(f"paper_metrics is not the independent test result: {path}")
    metrics = payload.get("test_metrics", {})
    errors.extend(confusion_errors(metrics, payload.get("class_names")))
    matrix = metrics.get("confusion_matrix")
    sample_count = metrics.get("sample_count")
    if not isinstance(matrix, list) or not matrix:
        errors.append(f"missing test confusion matrix: {path}")
    elif sum(sum(int(value) for value in row) for row in matrix) != sample_count:
        errors.append(f"confusion matrix total != test sample count: {path}")
    if not payload.get("run_signature"):
        errors.append(f"missing run signature: {path}")
    return errors


def validate_direct_screen(path: Path) -> list[str]:
    
    errors: list[str] = []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("complete") or not payload.get("run_signature"):
        errors.append(f"incomplete direct compressed screen: {path}")
    if payload.get("independent_test_opened") is not False:
        errors.append(f"direct screen opened the independent test: {path}")
    if payload.get("test_tensor_created") is not False:
        errors.append(f"direct screen created a test tensor: {path}")
    if payload.get("materialized_splits") != ["train", "val"]:
        errors.append(f"direct screen materialized unexpected splits: {path}")
    forbidden = {"test_metrics", "paper_metrics", "test_predictions"} & set(payload)
    if forbidden:
        errors.append(f"direct screen contains forbidden test outputs {sorted(forbidden)}: {path}")
    metrics = payload.get("validation_metrics", {})
    matrix = metrics.get("confusion_matrix")
    sample_count = metrics.get("sample_count")
    if not isinstance(matrix, list) or not matrix:
        errors.append(f"direct screen has no validation confusion matrix: {path}")
    elif sum(sum(int(value) for value in row) for row in matrix) != sample_count:
        errors.append(f"direct validation confusion matrix total mismatch: {path}")
    gate = payload.get("gate", {})
    checks = gate.get("checks", {})
    expected_checks = {
        "macro_f1_at_least_0.95",
        "loss_vs_no_cs_at_most_0.03",
        "every_class_recall_at_least_0.90",
    }
    if set(checks) != expected_checks:
        errors.append(f"direct screen gate checks are incomplete: {path}")
    if gate.get("passed") is not all(bool(value) for value in checks.values()):
        errors.append(f"direct screen gate/check mismatch: {path}")
    expected_status = "gate_passed" if gate.get("passed") else "gate_failed"
    if payload.get("status") != expected_status:
        errors.append(f"direct screen status/gate mismatch: {path}")
    history = payload.get("history", {})
    if int(payload.get("completed_epochs", -1)) != len(history.get("val_metrics", [])):
        errors.append(f"direct screen epoch-history mismatch: {path}")
    checkpoint = path.parent / str(payload.get("checkpoint_file", ""))
    if not checkpoint.is_file():
        errors.append(f"missing direct screen checkpoint: {checkpoint}")
    elif payload.get("checkpoint_sha256") != file_sha256(checkpoint):
        errors.append(f"direct screen checkpoint hash mismatch: {checkpoint}")
    config = payload.get("run_config", {})
    if config.get("materialized_splits") != ["train", "val"] or config.get("test_tensor_created") is not False:
        errors.append(f"direct screen run config violates validation-only scope: {path}")
    return errors


def validate_measurement_diagnostic_arm(path: Path) -> list[str]:
    
    errors: list[str] = []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("complete") or not payload.get("run_signature"):
        errors.append(f"incomplete measurement-diagnostic arm: {path}")
    config = payload.get("run_config", {})
    arm = config.get("arm")
    if arm not in {"raw_signal", "matched_measurements", "adjoint_lift"}:
        errors.append(f"unknown measurement-diagnostic arm: {path}")
    if payload.get("independent_test_opened") is not False or payload.get("test_tensor_created") is not False:
        errors.append(f"measurement-diagnostic arm opened the independent-test scope: {path}")
    if payload.get("materialized_splits") != ["train", "val"] or config.get("materialized_splits") != ["train", "val"]:
        errors.append(f"measurement-diagnostic arm materialized unexpected splits: {path}")
    forbidden = {"test_metrics", "paper_metrics", "test_predictions"} & set(payload)
    if forbidden:
        errors.append(f"measurement-diagnostic arm contains test outputs {sorted(forbidden)}: {path}")
    metrics = payload.get("validation_metrics", {})
    matrix = metrics.get("confusion_matrix")
    if not isinstance(matrix, list) or not matrix:
        errors.append(f"measurement-diagnostic arm has no validation confusion matrix: {path}")
    elif sum(sum(int(value) for value in row) for row in matrix) != metrics.get("sample_count"):
        errors.append(f"measurement-diagnostic arm confusion total mismatch: {path}")
    gate = payload.get("gate", {})
    checks = gate.get("checks", {})
    expected_checks = {
        "macro_f1_at_least_0.95",
        "loss_vs_no_cs_at_most_0.03",
        "every_class_recall_at_least_0.90",
    }
    if set(checks) != expected_checks or gate.get("passed") is not all(bool(value) for value in checks.values()):
        errors.append(f"measurement-diagnostic arm gate mismatch: {path}")
    expected_status = "gate_passed" if gate.get("passed") else "gate_failed"
    if payload.get("status") != expected_status:
        errors.append(f"measurement-diagnostic arm status/gate mismatch: {path}")
    history = payload.get("history", {})
    if int(payload.get("completed_epochs", -1)) != len(history.get("val_metrics", [])):
        errors.append(f"measurement-diagnostic arm epoch-history mismatch: {path}")
    checkpoint = path.parent / str(payload.get("checkpoint_file", ""))
    if not checkpoint.is_file():
        errors.append(f"missing measurement-diagnostic arm checkpoint: {checkpoint}")
    elif payload.get("checkpoint_sha256") != file_sha256(checkpoint):
        errors.append(f"measurement-diagnostic arm checkpoint hash mismatch: {checkpoint}")
    if arm == "raw_signal":
        if config.get("measurement_matrix") is not None or config.get("measurement_count") is not None:
            errors.append(f"raw-signal arm unexpectedly records a measurement matrix: {path}")
    elif not config.get("measurement_matrix_sha256") or not config.get("measurement_count"):
        errors.append(f"compressed arm lacks frozen measurement provenance: {path}")
    return errors


def validate_measurement_diagnostic_summary(path: Path) -> list[str]:
    
    errors: list[str] = []
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_arms = {"raw_signal", "matched_measurements", "adjoint_lift"}
    if not payload.get("complete") or payload.get("status") != "complete" or not payload.get("run_signature"):
        errors.append(f"incomplete measurement-domain diagnostic summary: {path}")
    if payload.get("independent_test_opened") is not False or payload.get("test_tensor_created") is not False:
        errors.append(f"measurement-domain summary opened the independent-test scope: {path}")
    if payload.get("materialized_splits") != ["train", "val"]:
        errors.append(f"measurement-domain summary materialized unexpected splits: {path}")
    arms = payload.get("arms", {})
    if set(arms) != expected_arms:
        errors.append(f"measurement-domain summary has incomplete arms: {path}")
        return errors
    for arm, metadata in arms.items():
        arm_path = Path(str(metadata.get("summary_file", "")))
        if not arm_path.is_file():
            errors.append(f"measurement-domain summary missing {arm} arm: {arm_path}")
            continue
        errors.extend(validate_measurement_diagnostic_arm(arm_path))
        arm_payload = json.loads(arm_path.read_text(encoding="utf-8"))
        if arm_payload.get("run_config", {}).get("arm") != arm:
            errors.append(f"measurement-domain summary/arm label mismatch: {path}")
        if metadata.get("run_signature") != arm_payload.get("run_signature"):
            errors.append(f"measurement-domain summary/arm signature mismatch: {path}")
        if metadata.get("validation_metrics") != arm_payload.get("validation_metrics"):
            errors.append(f"measurement-domain summary/arm metrics mismatch: {path}")
    return errors


def validate_task_aware_nested_screen(path: Path) -> list[str]:
    
    errors: list[str] = []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("complete") or not payload.get("run_signature"):
        errors.append(f"incomplete task-aware nested screen: {path}")
    if payload.get("independent_test_opened") is not False or payload.get("test_tensor_created") is not False:
        errors.append(f"task-aware nested screen opened the independent-test scope: {path}")
    if payload.get("materialized_splits") != ["train", "val"]:
        errors.append(f"task-aware nested screen materialized unexpected splits: {path}")
    forbidden = {"test_metrics", "paper_metrics", "test_predictions", "test_policy"} & set(payload)
    if forbidden:
        errors.append(f"task-aware nested screen contains forbidden test outputs {sorted(forbidden)}: {path}")
    config = payload.get("run_config", {})
    sensing = config.get("sensing", {})
    low_rows, high_rows = int(sensing.get("low_rows", 0)), int(sensing.get("high_rows", 0))
    signal_length = int(sensing.get("signal_length", 0))
    if sensing.get("matrix") != "learned binary +/-1/sqrt(N)" or not sensing.get("nested_prefix"):
        errors.append(f"task-aware nested sensing provenance is incomplete: {path}")
    if not 1 <= low_rows < high_rows <= signal_length:
        errors.append(f"task-aware nested row configuration is invalid: {path}")
    risk = config.get("risk_protocol", {})
    if risk.get("fit_scope") != "validation risk-fit source files only" or risk.get("threshold_scope") != "disjoint validation policy-select source files only":
        errors.append(f"task-aware nested risk protocol is incomplete: {path}")
    if set(risk.get("risk_fit_source_files", [])) & set(risk.get("policy_select_source_files", [])):
        errors.append(f"task-aware nested risk/policy sources overlap: {path}")
    if int(payload.get("validation_sample_count", -1)) != int(risk.get("risk_fit_sample_count", 0)) + int(risk.get("policy_select_sample_count", 0)):
        errors.append(f"task-aware nested validation partition count mismatch: {path}")
    history = payload.get("history", {})
    if len(history.get("train_loss", [])) != int(config.get("epochs", -1)):
        errors.append(f"task-aware nested training history length mismatch: {path}")
    if len(history.get("low_val_metrics", [])) != len(history.get("train_loss", [])) or len(history.get("high_val_metrics", [])) != len(history.get("train_loss", [])):
        errors.append(f"task-aware nested validation history mismatch: {path}")
    matrix_path = path.parent / str(payload.get("matrix_file", ""))
    if not matrix_path.is_file() or payload.get("matrix_sha256") != file_sha256(matrix_path):
        errors.append(f"task-aware nested matrix is missing or changed: {matrix_path}")
    else:
        try:
            with np.load(matrix_path, allow_pickle=False) as arrays:
                phi, low, high = arrays["phi"], arrays["low_prefix"], arrays["high_prefix"]
            expected = 1.0 / float(signal_length) ** 0.5
            if phi.shape != (high_rows, signal_length) or low.shape != (low_rows, signal_length) or high.shape != phi.shape:
                errors.append(f"task-aware nested matrix shapes mismatch: {matrix_path}")
            elif not np.all(np.isclose(np.abs(phi), expected, rtol=1e-6, atol=1e-7)):
                errors.append(f"task-aware nested matrix is not binary-normalized: {matrix_path}")
            elif not np.array_equal(low, phi[:low_rows]) or not np.array_equal(high, phi):
                errors.append(f"task-aware nested prefixes are not exact: {matrix_path}")
        except (KeyError, OSError, ValueError) as exc:
            errors.append(f"cannot read task-aware nested matrix: {matrix_path}: {exc}")
    model_path = path.parent / str(payload.get("model_checkpoint_file", ""))
    if not model_path.is_file() or payload.get("model_checkpoint_sha256") != file_sha256(model_path):
        errors.append(f"task-aware nested model checkpoint is missing or changed: {model_path}")
    risk_head = payload.get("risk_head", {})
    risk_path = path.parent / str(risk_head.get("checkpoint_file", ""))
    if not risk_path.is_file() or risk_head.get("checkpoint_sha256") != file_sha256(risk_path):
        errors.append(f"task-aware nested risk checkpoint is missing or changed: {risk_path}")
    selected = payload.get("validation_policy")
    expected_status = "gate_passed" if selected is not None else "gate_failed"
    if payload.get("status") != expected_status:
        errors.append(f"task-aware nested status/policy mismatch: {path}")
    if selected is not None:
        gate = selected.get("diagnostic_gate", {})
        if not gate.get("passed") or not all(bool(value) for value in gate.get("checks", {}).values()):
            errors.append(f"task-aware nested selected policy fails its gate: {path}")
        metrics = selected.get("metrics", {})
        matrix = metrics.get("confusion_matrix", [])
        if not matrix or sum(sum(int(value) for value in row) for row in matrix) != metrics.get("sample_count"):
            errors.append(f"task-aware nested policy confusion matrix mismatch: {path}")
    return errors


def validate_efficiency_summary(path: Path) -> list[str]:
    
    errors: list[str] = []
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_methods = {"no_cs", "ilsomp", "downsample", "pca", "random_projection"}
    if not payload.get("complete") or payload.get("status") != "complete":
        errors.append(f"incomplete efficiency summary: {path}")
    if payload.get("independent_test_opened") is not False:
        errors.append(f"efficiency benchmark opened the independent test: {path}")
    if not payload.get("protocol_signature") or not payload.get("claim_boundary"):
        errors.append(f"efficiency benchmark lacks provenance or claim boundary: {path}")
    protocol = payload.get("protocol", {})
    hardware = protocol.get("hardware", {})
    if not hardware.get("platform") or not hardware.get("torch"):
        errors.append(f"efficiency benchmark lacks hardware provenance: {path}")
    results = payload.get("results", {})
    if set(results) != expected_methods:
        errors.append(f"efficiency benchmark method set mismatch: {path}")
    for method, item in results.items():
        for metric_name in ("transform_latency_ms", "transform_plus_stft_latency_ms"):
            stats = item.get(metric_name, {})
            values = [stats.get("mean"), stats.get("median"), stats.get("p95")]
            if stats.get("n", 0) < 2 or any(value is None or not np.isfinite(value) or value < 0 for value in values):
                errors.append(f"invalid {method} {metric_name}: {path}")
        expected_measurements = 1024 if method == "no_cs" else 512
        if item.get("measurement_count") != expected_measurements:
            errors.append(f"unexpected measurement count for {method}: {path}")
    classifier = payload.get("classifier", {})
    if classifier.get("parameter_count", 0) <= 0 or classifier.get("latency_ms", {}).get("n", 0) < 2:
        errors.append(f"invalid classifier efficiency profile: {path}")
    offline_fit = payload.get("offline_fit", {})
    if offline_fit.get("pca_fit_seconds", -1) < 0 or offline_fit.get("pca_fit_sample_count", 0) <= 0:
        errors.append(f"invalid PCA offline-fit profile: {path}")
    return errors


def _is_diagnostic_artifact(path: Path, root: Path) -> bool:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return bool(relative.parts) and relative.parts[0] == "diagnostics"


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts_v2")
    cli.add_argument(
        "--extra-root",
        type=Path,
        action="append",
        default=[],
        help="additional artifacts_v2 roots, for example a D-drive continuation store",
    )
    cli.add_argument(
        "--include-diagnostics",
        action="store_true",
        help="also validate intentionally non-manuscript diagnostic artifacts",
    )
    args = cli.parse_args()
    roots = []
    for candidate in (args.root, *args.extra_root):
        resolved = candidate.resolve()
        if resolved not in roots:
            roots.append(resolved)
    manifests = [
        path for root in roots if root.exists() for path in root.rglob("dataset_manifest.json")
    ]
    metrics_files = [
        path for root in roots if root.exists() for path in root.rglob("*_metrics.json")
    ]
    
    
    
    direct_screens = (
        [path for root in roots if root.exists() for path in root.rglob("direct_compressed_validation_screen.json")]
    )
    measurement_domain_summaries = (
        [path for root in roots if root.exists() for path in root.rglob("measurement_domain_diagnostic_summary.json")]
    )
    task_aware_nested_screens = (
        [path for root in roots if root.exists() for path in root.rglob("task_aware_nested_validation_screen.json")]
    )
    efficiency_summaries = (
        [path for root in roots if root.exists() for path in root.rglob("efficiency_*_summary.json")]
    )
    if not args.include_diagnostics:
        manifests = [
            path for path in manifests
            if not any(_is_diagnostic_artifact(path, root) for root in roots)
        ]
        metrics_files = [
            path for path in metrics_files
            if not any(_is_diagnostic_artifact(path, root) for root in roots)
        ]
    errors = []
    for path in manifests:
        errors.extend(validate_manifest(path))
    for path in metrics_files:
        errors.extend(validate_metrics(path))
    for path in direct_screens:
        errors.extend(validate_direct_screen(path))
    for path in measurement_domain_summaries:
        errors.extend(validate_measurement_diagnostic_summary(path))
    for path in task_aware_nested_screens:
        errors.extend(validate_task_aware_nested_screen(path))
    for path in efficiency_summaries:
        errors.extend(validate_efficiency_summary(path))
    print(
        f"validated {len(manifests)} manifests, {len(metrics_files)} metric files, "
        f"{len(direct_screens)} direct diagnostic screens, and "
        f"{len(measurement_domain_summaries)} measurement-domain diagnostic summaries, and "
        f"{len(task_aware_nested_screens)} task-aware nested screens, and "
        f"{len(efficiency_summaries)} efficiency summaries"
    )
    archived = sum((path.parent / PRUNE_RECEIPT_NAME).is_file() for path in manifests)
    if archived:
        print(f"recognized {archived} archived rejected-diagnostic manifests")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    if not manifests:
        print("NOTICE: no v2 artifacts exist yet; run the clean study first")
    else:
        print("OK: artifact integrity checks passed")


if __name__ == "__main__":
    main()
