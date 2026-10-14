"""
Prompt-injection attack-type classifier. Every execution retrains from scratch.

 * Custom labels (main signal): the original pipeline method. Three HF datasets -> zero-shot teacher labels a seed
    sample (cached) -> TF-IDF student labels the rest (retrained each run) -> all_pseudo_labelled_attack_types.csv
 * Gold set (optional, training only): extra rows added at GOLD_WEIGHT.
 * Hold-out set (optional): hand-labelled prompts that are NEVER trained on; the only score of true accuracy.
Each run logs metrics to run_history.csv, saves training_dashboard.png, score_distributions.png and class_maps.png,
writes the most confident errors to confident_errors_*.csv, and saves the model (.joblib).
"""
import textwrap, json, joblib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from datetime import datetime
from pathlib import Path
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

SEED = 42
LABELS = ["Instruction Override", "System Prompt Extraction", "Indirect Prompt Injection", "Data Exfiltration",
        "Tool or Action Manipulation", "Jailbreak or Roleplay", "Other Attack"]
ZERO_SHOT = {  # label -> hypothesis description
    "Instruction Override": "An instruction override attack that tells the model to ignore, replace or destroy previous instructions",
    "System Prompt Extraction": "a system prompt extraction attack attempts to reveal hidden system instructions, developer instructions, or internal policies",
    "Indirect Prompt Injection": "an indirect prompt injection attack is hidden inside external content such as a website or email.",
    "Data Exfiltration": "a data exfiltration attack attempts to obtain secrets, private data, credentials, API keys, hidden content, memory, or confidential information",
    "Tool or Action Manipulation": "Tool or data manipulation attacks attempt to trigger unauthorised API calls, file actions, transactions, commands, or tool use",
    "Jailbreak or Roleplay": "a jailbreak or roleplay attack has the model assume the role or persona, or uses developer mode or simulations, to bypass restrictions",
    "Other Attack": "malicious attacks that do not fit the above categories"}

QUEUE, SEED_QUEUE = Path("attack_type_annotation_queue.csv"), Path("seed_attack_annotation_queue.csv")
SEED_LABELLED, STUDENT_LABELLED = Path("seed_automatically_labelled_attacks.csv"), Path("student_labelled_remaining_attacks.csv")
CUSTOM_LABELS_PATH = Path("all_pseudo_labelled_attack_types.csv")
GOLD_PATH, HOLDOUT_PATH = Path("gold_prompt_injection_set.csv"), Path("holdout_prompt_injection_set.csv")
MODEL_PATH, HISTORY_PATH = Path("gold_guided_attack_classifier.joblib"), Path("run_history.csv")
DASHBOARD, SCORES_PNG, MAPS_PNG, RUNS_DIR = Path("training_dashboard.png"), Path("score_distributions.png"), Path("class_maps.png"), Path("training_runs")
DATA_PATH, VAL_FRACTION = Path("cleaned.csv"), 0.2 
SEED_SIZE, GOLD_WEIGHT, THRESHOLD = 5000, 10.0, 0.5   # GOLD_WEIGHT: how much more a gold row counts; tune it
CURVE_FRACTIONS = [0.1, 0.25, 0.5, 0.75]              # the 100% point is the final model; [] skips the curve
MAP_POINTS = 3000                                     # max points drawn per scatter


# ============================================================ source datasets

def norm_text(s):
    return s.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

def load_source_datasets():
    """cleaned.csv (columns text, is_attack, source) -> stratified train / validation split with text, target, __source__."""
    raw = pd.read_csv(DATA_PATH)
    missing = {"text", "is_attack", "source"} - set(raw.columns)
    if missing:
        raise ValueError(f"{DATA_PATH} is missing columns: {sorted(missing)}")
    df = pd.DataFrame({"text": norm_text(raw["text"]), "target": raw["is_attack"].astype(int), "__source__": raw["source"]})
    df = df[df["text"] != ""].drop_duplicates("text").reset_index(drop=True)
    train, val = train_test_split(df, test_size=VAL_FRACTION, random_state=SEED, stratify=df["__source__"] + "|" + df["target"].astype(str))
    train, val = train.reset_index(drop=True), val.reset_index(drop=True)
    print(f"{DATA_PATH} -> train {len(train):,} ({(train.target == 0).sum():,} benign) | validation {len(val):,} ({(val.target == 0).sum():,} benign)")
    return train, val


# ============================================================ custom labels
def parse_labels(v):
    """'A|B' -> ['A','B']; 'BENIGN' or blank -> [] (benign = all-zero row)."""
    labels = [] if pd.isna(v) else [x.strip() for x in str(v).split("|") if x.strip()]
    return [] if labels == ["BENIGN"] else labels

def read_labelled(path, what):
    df = pd.read_csv(path)
    df["text"], df["labels"] = norm_text(df["text"]), df["custom_labels"].apply(parse_labels)
    unknown = set(df["labels"].explode().dropna()) - set(LABELS)
    if unknown:
        raise ValueError(f"Unknown labels in {what} ({path}): {sorted(unknown)}")
    return df

def zero_shot_label(input_path, output_path, model="typeform/distilbert-base-uncased-mnli", cutoff=0.55, batch=512):
    import torch
    from transformers import pipeline
    df = pd.read_csv(input_path)
    df["text"] = norm_text(df["text"])
    df = df[df["text"] != ""].reset_index(drop=True)
    device = 0 if torch.cuda.is_available() else -1
    clf = pipeline("zero-shot-classification", model=model, device=device)
    desc_to_label, texts, scores = {d: l for l, d in ZERO_SHOT.items()}, df["text"].tolist(), []
    print(f"ZERO-SHOT: {len(texts):,} prompts | {model} | {'GPU' if device == 0 else 'CPU'}")
    for i in range(0, len(texts), batch):
        res = clf(texts[i:i + batch], candidate_labels=list(ZERO_SHOT.values()), multi_label=True,
                hypothesis_template="This prompt is {}.", batch_size=32 if device == 0 else 8, truncation=True)
        scores += [{desc_to_label[d]: float(s) for d, s in zip(r["labels"], r["scores"])} for r in ([res] if isinstance(res, dict) else res)]
        print(f"Classified {len(scores):,} / {len(texts):,}")
        part = df.iloc[:len(scores)].copy()
        part["custom_labels"] = ["|".join(sorted(t for t, s in sc.items() if s >= cutoff) or ["Other Attack"]) for sc in scores]
        part.to_csv(str(output_path).replace(".csv", "_checkpoint.csv"), index=False)
    df["custom_labels"] = part["custom_labels"].values
    df["top_predicted_label"] = [max(sc, key=sc.get) for sc in scores]
    for t in ZERO_SHOT:
        df[f"{t}_score"] = [sc[t] for sc in scores]
    df["labelling_method"], df["label_threshold"] = f"zero_shot:{model}", cutoff
    df.to_csv(output_path, index=False)
    return df

def build_custom_labels(train, val):
    """queue -> zero-shot seed (cached) -> student (retrained every run) -> combined custom labels."""
    if not QUEUE.exists():
        att = pd.concat([train[train.target == 1].assign(split="train"), val[val.target == 1].assign(split="validation")], ignore_index=True)
        att[["text", "__source__", "split"]].assign(custom_labels="", annotation_notes="").to_csv(QUEUE, index=False)
    queue = pd.read_csv(QUEUE).reset_index(drop=True)
    if "row_id" not in queue:
        queue["row_id"] = queue.index
        queue.to_csv(QUEUE, index=False)
    print(f"Attack prompts in queue: {len(queue):,}")

    if not SEED_LABELLED.exists():
        queue.sample(n=min(SEED_SIZE, len(queue)), random_state=SEED).reset_index(drop=True).to_csv(SEED_QUEUE, index=False)
        zero_shot_label(SEED_QUEUE, SEED_LABELLED)
    else:
        print(f"Using cached zero-shot seed labels: {SEED_LABELLED}")
    seed = read_labelled(SEED_LABELLED, "seed labels")

    benign = pd.concat([train[train.target == 0][["text"]], val[val.target == 0][["text"]]])
    benign["labels"] = [[] for _ in range(len(benign))]
    stu = pd.concat([benign, seed[["text", "labels"]]], ignore_index=True).drop_duplicates("text")
    mlb = MultiLabelBinarizer(classes=LABELS)
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    print("Training the student on the zero-shot seed labels...")
    student = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED), n_jobs=-1)
    student.fit(vec.fit_transform(stu["text"]), mlb.fit_transform(stu["labels"]))

    rest = queue[~queue["row_id"].isin(set(seed["row_id"].astype(int)))].reset_index(drop=True)
    if len(rest):
        probs = student.predict_proba(vec.transform(rest["text"]))
        rest["labels"] = [list(p) or ["Other Attack"] for p in mlb.inverse_transform((probs >= THRESHOLD).astype(int))]
        rest["custom_labels"] = rest["labels"].apply("|".join)
        rest = rest.join(pd.DataFrame(probs, columns=[f"{c}_probability" for c in mlb.classes_]))
        rest.assign(labelling_method="student_tfidf_logistic_regression").to_csv(STUDENT_LABELLED, index=False)
    else:
        rest["labels"], rest["custom_labels"] = [], []
    cols = ["text", "__source__", "split", "row_id", "labels", "custom_labels"]
    combined = (pd.concat([seed[cols].assign(labelling_method="zero_shot_teacher"), rest[cols].assign(labelling_method="student_tfidf_logistic_regression")],
                        ignore_index=True).drop_duplicates("row_id").sort_values("row_id").reset_index(drop=True))
    combined.drop(columns="labels").to_csv(CUSTOM_LABELS_PATH, index=False)
    print(f"Custom labels: {len(combined):,} attacks ({len(seed):,} zero-shot, {len(rest):,} student)")
    return combined


# ============================================================ training / evaluation data
def build_base_sets(train, val, custom):
    attacks = set(train.loc[train.target == 1, "text"]) | set(val.loc[val.target == 1, "text"])
    missing = len(attacks - set(custom["text"]))
    print(f"Dataset attacks without a custom label: {missing:,} of {len(attacks):,}"
        + ("  <- delete the queue/seed/student CSVs to rebuild" if missing else ""))
    def assemble(df, split):
        b = df[df.target == 0][["text"]].copy()
        b["labels"] = [[] for _ in range(len(b))]
        return pd.concat([b, custom[custom["split"] == split][["text", "labels"]]], ignore_index=True)
    return assemble(train, "train"), assemble(val, "validation")

def load_optional(path, what):
    if not path.exists():
        print(f"No {what} at {path}.")
        return pd.DataFrame(columns=["text", "labels"])
    return read_labelled(path, what)[["text", "labels"]].drop_duplicates("text").reset_index(drop=True)

def drop_texts(df, texts):
    return df[~df["text"].isin(texts)].drop_duplicates("text").reset_index(drop=True)


# ============================================================ model
def fit_models(df, weights, mlb):
    """Word + char TF-IDF features, one logistic regression per label (so sample weights are honoured)."""
    feats = FeatureUnion([
        ("word", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=30000, sublinear_tf=True, min_df=2))])
    X, Y = feats.fit_transform(df["text"]), mlb.transform(df["labels"])
    models = [LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED).fit(X, Y[:, j], sample_weight=weights)
                if Y[:, j].sum() else None for j in range(Y.shape[1])]
    return feats, models

def predict_proba(feats, models, texts):
    X = feats.transform(texts)
    return np.column_stack([np.zeros(X.shape[0]) if m is None else m.predict_proba(X)[:, 1] for m in models])

def with_gold(base, gold):
    return pd.concat([base, gold], ignore_index=True), np.r_[np.ones(len(base)), np.full(len(gold), GOLD_WEIGHT)]

def fit_eval(train_df, weights, mlb, targets):
    """Fit once; return the model plus a score dict for each named evaluation frame."""
    feats, models = fit_models(train_df, weights, mlb)
    return feats, models, {k: score(predict_proba(feats, models, df["text"]), df, mlb) for k, df in targets.items() if len(df)}

def score(probs, df, mlb):
    y, pred = mlb.transform(df["labels"]), (probs >= THRESHOLD).astype(int)
    p, r, f, n = precision_recall_fscore_support(y, pred, zero_division=0)
    return {"y": y, "pred": pred, "probs": probs, "wrong": (pred != y).any(axis=1),
            "per_class": pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": n}, index=mlb.classes_),
            "micro": f1_score(y, pred, average="micro", zero_division=0), "macro": f1_score(y, pred, average="macro", zero_division=0),
            "attack": f1_score(y.any(axis=1), pred.any(axis=1), zero_division=0)}

def report(name, m, mlb):
    print(f"\n===== {name} =====")
    print(classification_report(m["y"], m["pred"], target_names=list(mlb.classes_), zero_division=0))
    print(f"Micro F1: {m['micro']:.4f} | Macro F1: {m['macro']:.4f} | Attack-vs-benign F1: {m['attack']:.4f}")


# ============================================================ visualisation
def palette_for(classes):
    return {"BENIGN": "#59A14F", **dict(zip(classes, sns.color_palette("husl", len(classes))))}

def save_fig(fig, path, stamp):
    RUNS_DIR.mkdir(exist_ok=True)
    for p in (path, RUNS_DIR / f"{path.stem}_{stamp}.png"):
        fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path} (copy in {RUNS_DIR}/)")

def eval_frame(res, base_val, holdout):
    """Hold-out (true accuracy) when available, otherwise the dataset validation split."""
    if len(holdout):
        return "HOLD-OUT (hand labels)", holdout, res["final_hold"], res["base_hold"]
    return "dataset validation (custom labels)", base_val, res["final_val"], res["base_val"]

def label_confusion(y, pred):
    """Rows = true label (plus 'no label'), columns = predicted label (plus 'none predicted').
    Cell (i, j) = how often rows truly carrying i were predicted j. A prompt with several labels counts in several rows."""
    t, p = np.c_[y, ~y.any(axis=1)].astype(int), np.c_[pred, ~pred.any(axis=1)].astype(int)
    counts = t.T @ p
    return counts, counts / np.maximum(t.sum(axis=0), 1)[:, None]

def plot_dashboard(res, curve, mlb, base_train, base_val, holdout, history, stamp):
    sns.set_theme(style="whitegrid", context="notebook")
    classes, has_hold = list(mlb.classes_), len(holdout) > 0
    pal, wrap = palette_for(classes), lambda c: textwrap.fill(c, 14, break_long_words=False)
    name, _, ev, ev_base = eval_frame(res, base_val, holdout)
    fig, axes = plt.subplots(2, 3, figsize=(22, 13))

    # 1. per-class precision vs recall, before and after adding the gold rows
    ax = axes[0, 0]
    for c in classes:
        pa, pb = ev_base["per_class"].loc[c], ev["per_class"].loc[c]
        ax.plot([pa.recall, pb.recall], [pa.precision, pb.precision], color=pal[c], alpha=0.35, lw=1)
        ax.scatter(pa.recall, pa.precision, s=90, facecolor="white", edgecolor=pal[c], linewidth=2, zorder=3)
        ax.scatter(pb.recall, pb.precision, s=130, color=pal[c], edgecolor="black", linewidth=0.8, zorder=4)
        ax.annotate(c, (pb.recall, pb.precision), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.scatter([], [], s=90, facecolor="white", edgecolor="black", linewidth=2, label="Custom labels only")
    ax.scatter([], [], s=130, color="grey", edgecolor="black", label="Custom labels + gold")
    ax.set(xlim=(-0.03, 1.05), ylim=(-0.03, 1.05), xlabel="Recall", ylabel="Precision", title=f"Per-class precision vs recall ({name})")
    ax.legend(loc="lower left", fontsize=9)

    # 2. label confusion matrix
    ax = axes[0, 1]
    counts, norm = label_confusion(ev["y"], ev["pred"])
    annot = np.array([[f"{norm[i, j]:.0%}\n({counts[i, j]})" for j in range(norm.shape[1])] for i in range(norm.shape[0])])
    sns.heatmap(norm, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=1, cbar=False, ax=ax, annot_kws={"fontsize": 8},
                xticklabels=[wrap(c) for c in classes] + ["none\npredicted"], yticklabels=[wrap(c) for c in classes] + ["BENIGN\n(no label)"])
    ax.tick_params(axis="x", labelsize=8, rotation=0); ax.tick_params(axis="y", labelsize=8, rotation=0)
    ax.set(xlabel="Model predicted", ylabel="True label", title=f"Label confusion: share of each true label predicted as each label ({name})")

    # 3. labels per prompt, true vs predicted (bubble size = number of prompts)
    ax = axes[0, 2]
    pairs = pd.Series(list(zip(ev["y"].sum(axis=1), ev["pred"].sum(axis=1)))).value_counts()
    px, py = zip(*pairs.index)
    ax.scatter(px, py, s=80 + 1500 * pairs.values / pairs.max(), alpha=0.6, color="#4C78A8", edgecolor="black")
    for (a, b), n in pairs.items():
        ax.annotate(str(n), (a, b), ha="center", va="center", fontsize=9, fontweight="bold")
    top = max(max(px), max(py)) + 0.7
    ax.plot([-0.5, top], [-0.5, top], "--", color="grey")
    ax.set(xlim=(-0.5, top), ylim=(-0.5, top), xticks=range(int(top)), yticks=range(int(top)), xlabel="True labels per prompt",
            ylabel="Predicted labels per prompt", title="Labels per prompt: above the line = over-labelling, below = under-labelling")

    # 4. learning curve
    ax = axes[1, 0]
    ax.scatter(curve["rows"], curve["val_macro"], s=110, label="Macro F1, dataset validation")
    if has_hold:
        ax.scatter(curve["rows"], curve["hold_macro"], s=130, marker="D", label="Macro F1, HOLD-OUT")
    ax.set(xscale="log", ylim=(0, 1.02), ylabel="F1", xlabel="Custom-labelled training rows used (gold rows always included)",
            title="Learning curve: model refitted on growing slices of the data")
    ax.legend()

    # 5. progress across executions
    ax, x = axes[1, 1], np.arange(1, len(history) + 1)
    for col, lab, mk in [("baseline_macro_f1", "Custom labels only, dataset-val macro F1", "o"),
                        ("final_macro_f1", "Custom labels + gold, dataset-val macro F1", "s"),
                        ("final_holdout_macro_f1", "Custom labels + gold, HOLD-OUT macro F1", "D")]:
        if history[col].notna().any():
            ax.scatter(x, history[col], s=100, marker=mk, label=lab, edgecolor="white", linewidth=0.6)
    ax.set(xticks=x, ylim=(0, 1.02), xlabel="Execution number", ylabel="F1", title=f"Progress across executions ({len(history)} so far)")
    ax.legend(loc="lower right", fontsize=8)

    # 6. per-class F1: agreement with the teacher vs true accuracy
    ax = axes[1, 2]
    if has_hold:
        fx, fy = res["final_val"]["per_class"]["f1"], res["final_hold"]["per_class"]["f1"]
        xl, yl, title = "F1 on dataset validation (agrees with custom labels)", "F1 on HOLD-OUT (hand labels)", \
            "Per-class F1: below the line = the custom labels flatter the model"
    else:
        fx, fy = res["base_val"]["per_class"]["f1"], res["final_val"]["per_class"]["f1"]
        xl, yl, title = "F1, custom labels only", "F1, custom labels + gold", "Per-class F1: above the line = gold helps"
    for c in classes:
        ax.scatter(fx[c], fy[c], s=150, color=pal[c], edgecolor="black", zorder=3)
        ax.annotate(c, (fx[c], fy[c]), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set(xlim=(-0.03, 1.05), ylim=(-0.03, 1.05), xlabel=xl, ylabel=yl, title=title)

    hold = f"HOLD-OUT macro F1 {res['final_hold']['macro']:.3f}  |  " if has_hold else ""
    fig.suptitle(f"Training dashboard  |  run {stamp}  |  {hold}vs custom labels: macro F1 {res['final_val']['macro']:.3f}  |  "
                f"custom-labelled rows {len(base_train):,}, hold-out rows {len(holdout):,}", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, DASHBOARD, stamp)

def plot_score_distributions(res, mlb, base_val, holdout, stamp):
    """Per class: the model's probability for prompts that do / do not truly carry the label. Dashed line = threshold."""
    sns.set_theme(style="whitegrid", context="notebook")
    classes, pal = list(mlb.classes_), palette_for(list(mlb.classes_))
    name, _, ev, _ = eval_frame(res, base_val, holdout)
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(ev["probs"]), min(MAP_POINTS, len(ev["probs"])), replace=False)
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    for j, c in enumerate(classes):
        ax, p_all, t_all = axes.flat[j], ev["probs"][:, j], ev["y"][:, j]
        auc = roc_auc_score(t_all, p_all) if 0 < t_all.sum() < len(t_all) else float("nan")
        p, t = p_all[idx], t_all[idx]
        wrong = (p >= THRESHOLD).astype(int) != t
        x = t + rng.uniform(-0.3, 0.3, len(t))
        for val, col in ((0, "#9AA0A6"), (1, pal[c])):
            m = (t == val) & ~wrong
            ax.scatter(x[m], p[m], s=22, alpha=0.6, color=col, edgecolor="white", linewidth=0.3)
        ax.scatter(x[wrong], p[wrong], s=55, marker="X", color="black", alpha=0.85, label="wrong at threshold")
        ax.axhline(THRESHOLD, ls="--", color="grey")
        ax.set(xlim=(-0.6, 1.6), ylim=(-0.03, 1.03), xticks=[0, 1], xticklabels=["does not\nhave label", "has label"],
                ylabel="Model probability", title=f"{c}\nAUC {auc:.2f} | {int(t_all.sum())} positives")
    last = axes.flat[len(classes)]
    last.axis("off")
    last.text(0, 0.5, f"Scored on: {name}\n\nEach dot is one prompt.\nWell separated clusters on either side of the\n"
            "dashed line mean the label is learnable.\nOverlap means no threshold can fix it.\n\n"
            "Black X = wrong at the threshold.\nAUC: 1.0 perfect separation, 0.5 chance.", va="center", fontsize=11)
    fig.suptitle(f"Score distributions per class  |  run {stamp}", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, SCORES_PNG, stamp)

def plot_class_maps(res, mlb, base_val, holdout, stamp):
    """Small multiples: every prompt placed by the model's own 7 probabilities (PCA to 2-D), one panel per class."""
    sns.set_theme(style="whitegrid", context="notebook")
    classes, pal = list(mlb.classes_), palette_for(list(mlb.classes_))
    name, _, ev, _ = eval_frame(res, base_val, holdout)
    rng, vp = np.random.RandomState(SEED), res["final_val"]["probs"]
    pca = PCA(n_components=2, random_state=SEED).fit(vp[rng.choice(len(vp), min(MAP_POINTS, len(vp)), replace=False)])
    var = pca.explained_variance_ratio_
    idx = rng.choice(len(ev["probs"]), min(MAP_POINTS, len(ev["probs"])), replace=False)
    co, y, pred = pca.transform(ev["probs"][idx]), ev["y"][idx], ev["pred"][idx]
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    for j, c in enumerate(classes):
        ax, pos, err = axes.flat[j], y[:, j] == 1, pred[:, j] != y[:, j]
        ax.scatter(co[:, 0], co[:, 1], s=14, color="#D0D3D8", alpha=0.6)
        ax.scatter(co[pos, 0], co[pos, 1], s=38, color=pal[c], edgecolor="white", linewidth=0.3)
        ax.scatter(co[err, 0], co[err, 1], s=50, marker="x", color="black", linewidth=1.2)
        ax.set(xlabel=f"Probability PC1 ({var[0]:.0%})", ylabel=f"Probability PC2 ({var[1]:.0%})",
                title=f"{c}\n{int(pos.sum())} true positives, {int(err.sum())} errors on this label")
    last = axes.flat[len(classes)]
    last.axis("off")
    last.text(0, 0.5, f"Scored on: {name}\n\nEach point is a prompt, placed by the\nmodel's seven output probabilities.\n\n"
            "Coloured = truly has this label.\nGrey = everything else.\nBlack x = wrong for this label.", va="center", fontsize=11)
    fig.suptitle(f"Class maps in probability space  |  run {stamp}", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, MAPS_PNG, stamp)

def write_confident_errors(df, m, mlb, path, n=200):
    """Wrong rows ranked by how confidently the model was wrong: the quickest way to find bad labels."""
    cls, rows = np.array(mlb.classes_), np.where(m["wrong"])[0]
    gap = (np.abs(m["probs"] - THRESHOLD) * (m["pred"] != m["y"])).max(axis=1)
    out = pd.DataFrame({"text": df["text"].iloc[rows].values,
                        "true_labels": ["|".join(cls[m["y"][i] == 1]) or "BENIGN" for i in rows],
                        "predicted": ["|".join(cls[m["pred"][i] == 1]) or "BENIGN" for i in rows],
                        "error_confidence": gap[rows]}).sort_values("error_confidence", ascending=False).head(n)
    out.to_csv(path, index=False)
    print(f"Saved {len(out)} most confident errors: {path}")


# ============================================================ main
def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"=== Training run {stamp} (everything is retrained from scratch) ===")
    train, val = load_source_datasets()
    base_train, base_val = build_base_sets(train, val, build_custom_labels(train, val))
    gold, holdout = load_optional(GOLD_PATH, "gold set"), load_optional(HOLDOUT_PATH, "hold-out set")
    if len(holdout):   # the hold-out set must never reach training
        hold = set(holdout["text"])
        removed = sum(d["text"].isin(hold).sum() for d in (base_train, base_val, gold))
        print(f"Hold-out set: {len(holdout):,} prompts, never trained on ({removed} matching training rows removed)")
        base_train, base_val, gold = (drop_texts(d, hold) for d in (base_train, base_val, gold))
    base_train, base_val = drop_texts(base_train, set(gold["text"])), drop_texts(base_val, set(gold["text"]) | set(base_train["text"]))
    print(f"Custom-labelled train {len(base_train):,} | dataset validation {len(base_val):,} | gold (train only) {len(gold):,} | hold-out (score only) {len(holdout):,}")

    mlb = MultiLabelBinarizer(classes=LABELS).fit([[]])
    targets = {"val": base_val, "hold": holdout}

    print("\nTraining on custom labels only...")
    _, _, base = fit_eval(base_train, np.ones(len(base_train)), mlb, targets)
    print("\nTraining on custom labels + gold rows...")
    mix, w = with_gold(base_train, gold)
    feats, _, final = fit_eval(mix, w, mlb, targets)
    for tag, d in (("CUSTOM LABELS ONLY", base), ("CUSTOM LABELS + GOLD", final)):
        report(f"{tag}, dataset validation", d["val"], mlb)
        if "hold" in d:
            report(f"{tag}, HOLD-OUT (true accuracy)", d["hold"], mlb)

    curve = []
    for frac in CURVE_FRACTIONS:
        print(f"\nLearning-curve fit at {frac:.0%} of the custom-labelled rows...")
        part = base_train.sample(frac=frac, random_state=SEED)
        _, _, d = fit_eval(*with_gold(part, gold), mlb, targets)
        curve.append({"rows": len(part), "val_macro": d["val"]["macro"], "hold_macro": d["hold"]["macro"] if "hold" in d else np.nan})
    curve.append({"rows": len(base_train), "val_macro": final["val"]["macro"], "hold_macro": final["hold"]["macro"] if "hold" in final else np.nan})
    curve = pd.DataFrame(curve)

    nan = lambda d, k, f: d[k][f] if k in d else np.nan
    record = pd.DataFrame([{
        "run": stamp, "custom_labelled_train_rows": len(base_train), "dataset_val_rows": len(base_val), "gold_rows": len(gold),
        "gold_weight": GOLD_WEIGHT, "holdout_rows": len(holdout),
        "baseline_macro_f1": base["val"]["macro"], "baseline_micro_f1": base["val"]["micro"],
        "final_macro_f1": final["val"]["macro"], "final_micro_f1": final["val"]["micro"], "final_attack_vs_benign_f1": final["val"]["attack"],
        "baseline_holdout_macro_f1": nan(base, "hold", "macro"), "final_holdout_macro_f1": nan(final, "hold", "macro"),
        "final_holdout_micro_f1": nan(final, "hold", "micro"), "final_holdout_attack_vs_benign_f1": nan(final, "hold", "attack"),
        "per_class_f1": json.dumps({k: round(float(v), 3) for k, v in final["val"]["per_class"]["f1"].items()}),
        "holdout_per_class_f1": json.dumps({k: round(float(v), 3) for k, v in final["hold"]["per_class"]["f1"].items()}) if "hold" in final else ""}])
    history = pd.concat([pd.read_csv(HISTORY_PATH), record], ignore_index=True) if HISTORY_PATH.exists() else record
    history.to_csv(HISTORY_PATH, index=False)
    print(f"Logged run to {HISTORY_PATH}")
    res = {"base_val": base["val"], "final_val": final["val"], "base_hold": base.get("hold"), "final_hold": final.get("hold")}
    plot_dashboard(res, curve, mlb, base_train, base_val, holdout, history, stamp)
    plot_score_distributions(res, mlb, base_val, holdout, stamp)
    plot_class_maps(res, mlb, base_val, holdout, stamp)
    write_confident_errors(base_val, final["val"], mlb, Path("confident_errors_dataset_validation.csv"))
    if "hold" in final:
        write_confident_errors(holdout, final["hold"], mlb, Path("confident_errors_holdout.csv"))

    print("\nTraining production model on all custom-labelled data + gold rows...")
    feats_f, models_f = fit_models(*with_gold(pd.concat([base_train, base_val], ignore_index=True), gold), mlb)
    joblib.dump({"features": feats_f, "classifiers": models_f, "attack_type_labels": list(mlb.classes_), "threshold": THRESHOLD,
                "gold_weight": GOLD_WEIGHT, "trained_at": stamp, "holdout_macro_f1": nan(final, "hold", "macro"),
                "trained_on": [str(DATA_PATH), str(CUSTOM_LABELS_PATH), str(GOLD_PATH)],
                "model_type": "custom_label_word_char_tfidf_per_label_logistic_regression"}, MODEL_PATH)
    print(f"Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()