import numpy as np


def confusion_errors(metrics, class_names=None):
    errors = []
    try:
        if not {"accuracy", "macro_f1"}.issubset(metrics):
            return ["missing accuracy or macro-F1"]
        raw = np.asarray(metrics["confusion_matrix"])
        if raw.ndim != 2 or raw.shape[0] != raw.shape[1] or raw.size == 0:
            return ["confusion matrix must be nonempty and square"]
        if not np.all(np.isfinite(raw)) or np.any(raw < 0) or not np.all(raw == np.floor(raw)):
            return ["confusion counts must be finite nonnegative integers"]
        matrix = raw.astype(np.float64)
        total = matrix.sum()
        if total <= 0 or total != metrics["sample_count"]:
            return ["confusion total differs from positive sample count"]
        hits = np.diag(matrix)
        recalls = np.divide(hits, matrix.sum(1), out=np.zeros(len(hits)), where=matrix.sum(1) > 0)
        precisions = np.divide(hits, matrix.sum(0), out=np.zeros(len(hits)), where=matrix.sum(0) > 0)
        den = matrix.sum(0) + matrix.sum(1)
        f1 = np.divide(2 * hits, den, out=np.zeros(len(hits)), where=den > 0)
        expected = {"accuracy": hits.sum() / total, "macro_f1": f1.mean(),
                    "macro_precision": precisions.mean(), "macro_recall": recalls.mean()}
        for name, value in expected.items():
            if name in metrics and not np.isclose(metrics[name], value, atol=1e-10, rtol=0):
                errors.append(f"{name} differs from confusion-matrix recomputation")
        if class_names is not None:
            if len(class_names) != len(hits) or len(set(class_names)) != len(hits):
                errors.append("class order or count is invalid")
            for key in ("per_class_recall", "per_class_recalls"):
                if key in metrics:
                    mapping = metrics[key]
                    if set(mapping) != set(class_names) or any(not np.isclose(mapping[name], recalls[i], atol=1e-10, rtol=0) for i, name in enumerate(class_names)):
                        errors.append("per-class recalls differ from confusion matrix")
    except (KeyError, TypeError, ValueError):
        errors.append("invalid or incomplete metric payload")
    return errors


def statistic_errors(saved, values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return ["invalid aggregate inputs"]
    expected = {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else None}
    if "n" in saved:
        expected["n"] = len(values)
    errors = []
    for name, value in expected.items():
        if value is None:
            if saved.get(name) is not None:
                errors.append(f"aggregate {name} differs")
        elif name not in saved or not np.isclose(saved[name], value, atol=1e-10, rtol=0):
            errors.append(f"aggregate {name} differs")
    return errors
