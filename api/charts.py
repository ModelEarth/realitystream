"""
Report charts: PNGs in a run's report folder, which its index.html (localsite's displayCharts()) shows.

    metrics_dashboard.png           accuracy, ROC-AUC, F1, precision, recall and balanced accuracy per model
    metrics_comparison.png          the same six metrics, one panel each (3 x 2), with each bar's value
    roc_curves_comparison.png       each model's ROC curve (binary targets)
    confusion_matrices.png          one matrix per model
    training_time_comparison.png    training seconds per model
    feature_importance_{model}.png  top features: rfc, xgboost and lr natively, svm and mlp by permutation

run_models.run_pipeline calls write_charts() with one training pass (with SMOTE when it ran).
"""

import os

import numpy as np

DPI = 100
TOP_FEATURES = 15
NAME_CHARS = 45  # Long feature names (NAICS titles) are cut to this length
# Short names and fixed colors, so each model looks the same in every chart
SHORT_NAMES = {"lr": "LR", "rfc": "RFC", "rbf": "RBF", "svm": "SVM", "mlp": "MLP", "xgboost": "XGBoost"}
COLORS = {"lr": "#4c78a8", "rfc": "#59a14f", "rbf": "#e15759", "svm": "#f28e2b", "mlp": "#b07aa1", "xgboost": "#76b7b2"}
METRICS = [("accuracy", "Accuracy"), ("roc_auc", "ROC-AUC"), ("f1_score", "F1"), ("precision", "Precision"),
           ("recall", "Recall"), ("balanced_accuracy", "Balanced acc.")]


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")  # No display on Cloud Run
    import matplotlib.pyplot as plt
    return plt


def _name(key):
    return SHORT_NAMES.get(key, key)


def _color(key):
    return COLORS.get(key, "#888888")


def metrics_dashboard(plt, results, y_true, label):
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / len(results)
    x = np.arange(len(METRICS))
    for i, r in enumerate(results):
        values = [r.get(field) or 0 for field, _ in METRICS]
        ax.bar(x + (i - (len(results) - 1) / 2) * width, values, width, label=_name(r["model_type"]),
               color=_color(r["model_type"]))
    ax.set_xticks(x, [title for _, title in METRICS])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Model metrics{label}")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    return fig


def metrics_comparison(plt, results, y_true, label):
    """One panel per metric, like the 2025 Colab's Eye Blinks dashboard, with all six metrics."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    names = [_name(r["model_type"]) for r in results]
    colors = [_color(r["model_type"]) for r in results]
    for ax, (field, title) in zip(axes.flat, METRICS):
        values = [r.get(field) for r in results]
        bars = ax.bar(names, [v or 0 for v in values], color=colors, alpha=0.8, edgecolor="black", linewidth=1.2)
        for bar, value in zip(bars, values):
            if value is not None:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.4f}",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_ylabel(title)
        ax.set_title(f"{title} comparison", fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Model metrics{label}", fontsize=14)
    fig.tight_layout()
    return fig


def roc_curves(plt, results, y_true, label):
    from sklearn.metrics import auc, roc_curve

    if len(np.unique(y_true)) != 2:
        return None
    curves = [r for r in results if r.get("y_score") is not None]
    if not curves:
        return None
    fig, ax = plt.subplots(figsize=(7, 6))
    positive = np.unique(y_true)[1]
    for r in curves:
        fpr, tpr, _ = roc_curve(y_true, r["y_score"], pos_label=positive)
        ax.plot(fpr, tpr, color=_color(r["model_type"]), lw=2,
                label=f"{_name(r['model_type'])} (AUC {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#999999", lw=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC curves{label}")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    return fig


def confusion_matrices(plt, results, y_true, label):
    from sklearn.metrics import confusion_matrix

    shown = [r for r in results if r.get("y_pred") is not None]
    if not shown:
        return None
    classes = np.unique(np.concatenate([y_true] + [np.asarray(r["y_pred"]).ravel() for r in shown]))
    cols = min(3, len(shown))
    rows = -(-len(shown) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows), squeeze=False)
    for ax, r in zip(axes.flat, shown):
        matrix = confusion_matrix(y_true, r["y_pred"], labels=classes)
        ax.imshow(matrix, cmap="Blues", vmin=0)  # From 0, so the text color below matches the shade
        for (i, j), count in np.ndenumerate(matrix):
            ax.text(j, i, str(count), ha="center", va="center",
                    color="white" if count > matrix.max() / 2 else "black")
        ax.set_xticks(range(len(classes)), classes)
        ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{_name(r['model_type'])}, accuracy {r['accuracy']:.3f}")
    for ax in list(axes.flat)[len(shown):]:
        ax.axis("off")
    fig.suptitle(f"Confusion matrices{label}")
    fig.tight_layout()
    return fig


def training_times(plt, results, y_true, label):
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(results) + 1.5))
    names = [_name(r["model_type"]) for r in results]
    seconds = [r.get("time") or 0 for r in results]
    bars = ax.barh(names, seconds, color=[_color(r["model_type"]) for r in results])
    ax.bar_label(bars, labels=[f"{s:.1f} s" for s in seconds], padding=3, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Training seconds (XGBoost and MLP include their hyperparameter search)")
    ax.set_title(f"Training time{label}")
    ax.margins(x=0.15)
    return fig


def feature_importance(plt, key, title, df, label):
    top = df.head(TOP_FEATURES).iloc[::-1]
    names = [str(f) if len(str(f)) <= NAME_CHARS else str(f)[:NAME_CHARS - 1] + "…" for f in top["Feature"]]
    fig, ax = plt.subplots(figsize=(9, 0.35 * len(top) + 1.5))
    ax.barh(names, top["Importance"], color=_color(key))
    method = df.attrs.get("method", "")
    ax.set_xlabel("Drop in score when the feature is shuffled" if method == "permutation" else "Importance")
    ax.set_title(f"{title}: top features{label}")
    return fig


CHARTS = [
    ("metrics_dashboard.png", metrics_dashboard),
    ("metrics_comparison.png", metrics_comparison),
    ("roc_curves_comparison.png", roc_curves),
    ("confusion_matrices.png", confusion_matrices),
    ("training_time_comparison.png", training_times),
]


def write_charts(report_dir, results, y_test, importances=None, titles=None, label=""):
    """Write the PNG charts for one training pass's results. Returns the file names written.
    A chart that fails is left out with a warning, so the report keeps the others."""
    if not results:
        return []
    plt = _pyplot()
    y_true = np.asarray(y_test).ravel()
    label = f" ({label})" if label else ""
    drawn = [(name, lambda draw=draw: draw(plt, results, y_true, label)) for name, draw in CHARTS]
    for key, df in (importances or {}).items():
        title = (titles or {}).get(key, _name(key))
        drawn.append((f"feature_importance_{key}.png",
                      lambda key=key, title=title, df=df: feature_importance(plt, key, title, df, label)))
    written = []
    for name, draw in drawn:
        try:
            fig = draw()
        except Exception as exc:
            print(f"  [WARN] {name} not drawn: {exc}")
            plt.close("all")
            continue
        if fig is None:
            continue
        fig.savefig(os.path.join(report_dir, name), dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        written.append(name)
    print(f"  [CHARTS] {', '.join(written) or 'none'}")
    return written
