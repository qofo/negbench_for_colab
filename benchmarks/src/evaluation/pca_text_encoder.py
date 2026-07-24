"""
CLIP Negation Information Degradation Mechanism Analysis Module (Refined 3rd Edition).

Stage 1 Experiments (Top Priority):
  1. Pipeline 5-Step Degradation Tracking (Line Plot)
  2. Direction Preservation Analysis (Distance Norm Ratio: Negation vs Random Control)
  3. Linear Probe Analysis (LogisticRegression 5-Fold CV Accuracy: Pre vs Post Projection)
  4. Representation Geometry & Intrinsic Dimensionality Analysis
     - Effective Rank (r_eff) & Participation Ratio (PR) Pre vs Post Projection
     - PCA Variance Spectrum & Anisotropy Analysis

Stage 2 & 3 Experiments:
  5. Text-side Projection Causal Ablation (Original W_proj vs Identity vs Random Orthogonal)
  6. Retrieval Metrics & Ranking Flip Rate (Accuracy, MRR, Flip Rate)
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any

import open_clip


# ==============================================================================
# Granular 5-Step Pipeline Feature Extraction
# ==============================================================================

def extract_pipeline_step_features(
    model: nn.Module,
    tokenizer: Any,
    texts: List[str],
    device: str = "cpu",
    batch_size: int = 256,
    custom_projection: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Extract features at 5 granular pipeline steps matching exact CLIP execution order:
      Step 0: 'Embedding' (Token + Positional Embedding EOT)
      Step 1: 'Layer12_Raw' (Layer 12 output EOT before ln_final)
      Step 2: 'Layer12_LN' (Layer 12 output EOT after ln_final)
      Step 3: 'Projected_Unnorm' (after text_projection EOT)
      Step 4: 'Final_L2Norm' (after L2 normalization)
    """
    model.eval()
    all_tokens = tokenizer(texts).to(device)

    text_tower = getattr(model, 'text', model)
    token_embedding = text_tower.token_embedding
    positional_embedding = text_tower.positional_embedding
    transformer = text_tower.transformer
    ln_final = text_tower.ln_final
    text_projection = getattr(text_tower, 'text_projection', None)
    attn_mask = getattr(text_tower, 'attn_mask', None)

    steps_data = {
        "Step0_Embedding": [],
        "Step1_Layer12_Raw": [],
        "Step2_Layer12_LN": [],
        "Step3_Projected_Unnorm": [],
        "Step4_Final_L2Norm": []
    }

    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch_tokens = all_tokens[start:end]

        with torch.no_grad():
            cast_dtype = transformer.get_cast_dtype()
            eot_indices = batch_tokens.argmax(dim=-1).cpu()
            batch_idx = torch.arange(batch_tokens.shape[0])

            # Step 0: Token + Positional Embedding
            x = token_embedding(batch_tokens).to(cast_dtype)
            seq_len = batch_tokens.shape[1]
            x = x + positional_embedding[:seq_len].to(cast_dtype)
            step0 = x[batch_idx, eot_indices].float().cpu()

            # Step 1: Layer 12 Raw output (before LN)
            x_perm = x.permute(1, 0, 2)
            x_trans = transformer(x_perm, attn_mask=attn_mask)
            x_trans = x_trans.permute(1, 0, 2)
            step1 = x_trans[batch_idx, eot_indices].float().cpu()

            # Step 2: Layer 12 + LN (after ln_final)
            x_ln = ln_final(x_trans)
            step2 = x_ln[batch_idx, eot_indices].float().cpu()

            # Step 3: Projection
            if custom_projection is not None:
                if isinstance(custom_projection, str) and custom_projection == "identity":
                    step3 = step2.clone()
                else:
                    W_custom = torch.from_numpy(custom_projection).float().cpu()
                    step3 = step2 @ W_custom
            elif text_projection is not None:
                if isinstance(text_projection, nn.Linear):
                    step3 = text_projection(x_ln[batch_idx, eot_indices].to(text_projection.weight.dtype)).float().cpu()
                else:
                    step3 = (x_ln[batch_idx, eot_indices].to(text_projection.dtype) @ text_projection).float().cpu()
            else:
                step3 = step2.clone()

            # Step 4: L2 Normalization
            step4 = F.normalize(step3, dim=-1).cpu()

            steps_data["Step0_Embedding"].append(step0)
            steps_data["Step1_Layer12_Raw"].append(step1)
            steps_data["Step2_Layer12_LN"].append(step2)
            steps_data["Step3_Projected_Unnorm"].append(step3)
            steps_data["Step4_Final_L2Norm"].append(step4)

    return {k: torch.cat(v, dim=0).numpy() for k, v in steps_data.items()}


# ==============================================================================
# Stage 1-A: Pipeline Step Breakdown (Line Plot)
# ==============================================================================

def analyze_pipeline_breakdown(
    model: nn.Module,
    tokenizer: Any,
    pos_texts: List[str],
    neg_texts: List[str],
    output_dir: str,
    device: str = "cpu",
    batch_size: int = 256,
) -> Dict[str, Any]:
    """Track cosine similarity step-by-step using a clean line plot."""
    import pandas as pd

    print("\n" + "="*60)
    print("Stage 1-A: Pipeline Step-by-Step Breakdown Analysis")
    print("="*60)

    pos_steps = extract_pipeline_step_features(model, tokenizer, pos_texts, device, batch_size)
    neg_steps = extract_pipeline_step_features(model, tokenizer, neg_texts, device, batch_size)

    step_names = [
        "Step0_Embedding",
        "Step1_Layer12_Raw",
        "Step2_Layer12_LN",
        "Step3_Projected_Unnorm",
        "Step4_Final_L2Norm"
    ]

    labels_map = {
        "Step0_Embedding": "Step 0: Embedding",
        "Step1_Layer12_Raw": "Step 1: Layer12 Raw",
        "Step2_Layer12_LN": "Step 2: Layer12+LN",
        "Step3_Projected_Unnorm": "Step 3: +Projection",
        "Step4_Final_L2Norm": "Step 4: +L2 Norm (Final)"
    }

    breakdown_results = []
    for idx, sname in enumerate(step_names):
        pos_f = pos_steps[sname]
        neg_f = neg_steps[sname]

        pos_norm = pos_f / (np.linalg.norm(pos_f, axis=1, keepdims=True) + 1e-8)
        neg_norm = neg_f / (np.linalg.norm(neg_f, axis=1, keepdims=True) + 1e-8)
        sims = np.sum(pos_norm * neg_norm, axis=1)

        mean_sim = float(np.mean(sims))
        median_sim = float(np.median(sims))
        std_sim = float(np.std(sims))

        breakdown_results.append({
            "step_id": idx,
            "step_key": sname,
            "step_name": labels_map[sname],
            "mean_cosine_sim": mean_sim,
            "median_cosine_sim": median_sim,
            "std_cosine_sim": std_sim,
        })
        print(f"  [{labels_map[sname]}] Mean Cosine Sim: {mean_sim:.4f} (Median: {median_sim:.4f})")

    df_breakdown = pd.DataFrame(breakdown_results)
    csv_path = os.path.join(output_dir, "pipeline_step_breakdown.csv")
    df_breakdown.to_csv(csv_path, index=False)

    # Line Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    x_labels = [labels_map[k] for k in step_names]
    means = df_breakdown["mean_cosine_sim"].values

    ax.plot(x_labels, means, "o-", color="crimson", lw=2.5, ms=8, label="Mean Cosine Sim")
    ax.set_ylabel("Positive↔Negative Cosine Similarity", fontsize=11)
    ax.set_title("Pipeline Breakdown: Representation Geometry Shift Across Pipeline", fontsize=12, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(fontsize=10)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "pipeline_step_lineplot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot_path}")

    return {"csv_path": csv_path, "plot_path": plot_path, "step_results": breakdown_results}


# ==============================================================================
# Stage 1-B: Direction Preservation Analysis (Distance Ratio: Negation vs Control)
# ==============================================================================

def analyze_direction_preservation(
    model: nn.Module,
    tokenizer: Any,
    pos_texts: List[str],
    neg_texts: List[str],
    output_dir: str,
    device: str = "cpu",
    batch_size: int = 256,
) -> Dict[str, Any]:
    """
    Compare Distance Ratio (||pos - neg||_post / ||pos - neg||_pre) for Negation vs Random Control.
    """
    import pandas as pd
    from scipy import stats

    print("\n" + "="*60)
    print("Stage 1-B: Direction Preservation Analysis (Negation vs Random Control)")
    print("="*60)

    pos_steps = extract_pipeline_step_features(model, tokenizer, pos_texts, device, batch_size)
    neg_steps = extract_pipeline_step_features(model, tokenizer, neg_texts, device, batch_size)

    pos_pre = pos_steps["Step2_Layer12_LN"]
    neg_pre = neg_steps["Step2_Layer12_LN"]

    pos_post = pos_steps["Step4_Final_L2Norm"]
    neg_post = neg_steps["Step4_Final_L2Norm"]

    dist_pre_neg = np.linalg.norm(pos_pre - neg_pre, axis=1)
    dist_post_neg = np.linalg.norm(pos_post - neg_post, axis=1)
    ratio_neg = dist_post_neg / (dist_pre_neg + 1e-8)

    np.random.seed(42)
    rand_idx = np.random.permutation(len(pos_texts))
    rand_pre = pos_pre[rand_idx]
    rand_post = pos_post[rand_idx]

    dist_pre_ctrl = np.linalg.norm(pos_pre - rand_pre, axis=1)
    dist_post_ctrl = np.linalg.norm(pos_post - rand_post, axis=1)
    ratio_ctrl = dist_post_ctrl / (dist_pre_ctrl + 1e-8)

    t_stat, p_val = stats.ttest_ind(ratio_neg, ratio_ctrl)

    print(f"Negation Pairs  : Mean Pre Dist={np.mean(dist_pre_neg):.4f} -> Post={np.mean(dist_post_neg):.4f} (Ratio={np.mean(ratio_neg):.4f})")
    print(f"Control Pairs   : Mean Pre Dist={np.mean(dist_pre_ctrl):.4f} -> Post={np.mean(dist_post_ctrl):.4f} (Ratio={np.mean(ratio_ctrl):.4f})")
    print(f"T-test Difference: t={t_stat:.4f}, p-value={p_val:.2e}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ratio_neg, bins=35, alpha=0.6, color="crimson", edgecolor="black", label=f"Negation Pairs (Mean Ratio: {np.mean(ratio_neg):.4f})")
    ax.hist(ratio_ctrl, bins=35, alpha=0.6, color="gray", edgecolor="black", label=f"Control Random Pairs (Mean Ratio: {np.mean(ratio_ctrl):.4f})")
    ax.set_title(f"Direction Preservation: Negation vs Control Pairs (p={p_val:.1e})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Distance Ratio (Post-Proj Dist / Pre-Proj Dist)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, ls="--", alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "direction_preservation_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    report = {
        "negation_mean_dist_pre": float(np.mean(dist_pre_neg)),
        "negation_mean_dist_post": float(np.mean(dist_post_neg)),
        "negation_mean_ratio": float(np.mean(ratio_neg)),
        "control_mean_dist_pre": float(np.mean(dist_pre_ctrl)),
        "control_mean_dist_post": float(np.mean(dist_post_ctrl)),
        "control_mean_ratio": float(np.mean(ratio_ctrl)),
        "ttest_t_stat": float(t_stat),
        "ttest_p_value": float(p_val)
    }

    rpt_path = os.path.join(output_dir, "direction_preservation_report.json")
    with open(rpt_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


# ==============================================================================
# Stage 1-C: Linear Probe Analysis (LogisticRegression Pre vs Post Projection)
# ==============================================================================

def analyze_linear_probe(
    model: nn.Module,
    tokenizer: Any,
    pos_texts: List[str],
    neg_texts: List[str],
    output_dir: str,
    device: str = "cpu",
    batch_size: int = 256,
) -> Dict[str, Any]:
    """
    Train a LogisticRegression linear probe on positive (1) vs negative (0) embeddings
    at Step 2 (Layer12+LN) vs Step 4 (Final L2 Norm) to measure linear separability.
    """
    print("\n" + "="*60)
    print("Stage 1-C: Linear Probe Analysis (Linear Separability Pre vs Post Projection)")
    print("="*60)

    pos_steps = extract_pipeline_step_features(model, tokenizer, pos_texts, device, batch_size)
    neg_steps = extract_pipeline_step_features(model, tokenizer, neg_texts, device, batch_size)

    n_pos = len(pos_texts)
    n_neg = len(neg_texts)
    y = np.array([1] * n_pos + [0] * n_neg)

    probe_results = {}
    step_keys = ["Step0_Embedding", "Step2_Layer12_LN", "Step4_Final_L2Norm"]
    step_labels = ["Step 0 (Embed)", "Step 2 (Layer12+LN)", "Step 4 (Final L2Norm)"]

    for skey, slabel in zip(step_keys, step_labels):
        X_pos = pos_steps[skey]
        X_neg = neg_steps[skey]
        X = np.vstack([X_pos, X_neg])

        clf = LogisticRegression(max_iter=1000, random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")

        mean_acc = float(np.mean(scores)) * 100
        std_acc = float(np.std(scores)) * 100
        probe_results[slabel] = {"mean_accuracy": mean_acc, "std_accuracy": std_acc}
        print(f"  [{slabel}] Linear Probe 5-Fold Accuracy: {mean_acc:.2f}% (±{std_acc:.2f}%)")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(probe_results.keys(), [v["mean_accuracy"] for v in probe_results.values()],
                  color=["gray", "seagreen", "crimson"], alpha=0.85, edgecolor="black")
    ax.set_ylabel("Linear Probe Accuracy (%)", fontsize=11)
    ax.set_title("Linear Probe: Linear Separability Pre vs Post Projection", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", ls="--", alpha=0.3)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "linear_probe_accuracy.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    report_path = os.path.join(output_dir, "linear_probe_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(probe_results, f, indent=2)

    return probe_results


# ==============================================================================
# Stage 1-D: Intrinsic Dimensionality & PCA Spectrum (Effective Rank, PR)
# ==============================================================================

def compute_intrinsic_dimensionality(X: np.ndarray) -> Tuple[float, float]:
    """
    Compute Effective Rank (r_eff) and Participation Ratio (PR) of feature matrix X.
      - Effective Rank: r_eff = exp(-sum p_i ln p_i) where p_i = lambda_i / sum lambda_k
      - Participation Ratio: PR = (sum lambda_i)^2 / sum(lambda_i^2)
    """
    X_centered = X - np.mean(X, axis=0, keepdims=True)
    cov = (X_centered.T @ X_centered) / X_centered.shape[0]
    eigenvals = np.linalg.eigvalsh(cov)
    eigenvals = np.sort(np.maximum(eigenvals, 1e-12))[::-1]  # Sort descending

    # Normalized probabilities p_i
    total_val = np.sum(eigenvals)
    p = eigenvals / total_val

    # 1. Effective Rank via Spectral Entropy
    entropy = -np.sum(p * np.log(p + 1e-12))
    eff_rank = float(np.exp(entropy))

    # 2. Participation Ratio
    pr = float((np.sum(eigenvals) ** 2) / np.sum(eigenvals ** 2))

    return eff_rank, pr


def analyze_pca_spectrum_compression(
    model: nn.Module,
    tokenizer: Any,
    pos_texts: List[str],
    neg_texts: List[str],
    output_dir: str,
    device: str = "cpu",
    batch_size: int = 256,
) -> Dict[str, Any]:
    """
    Compare PCA Spectrum & Intrinsic Dimension (Effective Rank & Participation Ratio)
    Pre-projection (Layer12+LN) vs Post-projection (Final L2Norm).
    """
    import pandas as pd

    print("\n" + "="*60)
    print("Stage 1-D: Intrinsic Dimensionality & PCA Spectrum Analysis")
    print("="*60)

    pos_steps = extract_pipeline_step_features(model, tokenizer, pos_texts, device, batch_size)
    neg_steps = extract_pipeline_step_features(model, tokenizer, neg_texts, device, batch_size)

    X_pre = np.vstack([pos_steps["Step2_Layer12_LN"], neg_steps["Step2_Layer12_LN"]])
    X_post = np.vstack([pos_steps["Step4_Final_L2Norm"], neg_steps["Step4_Final_L2Norm"]])

    # Intrinsic Dimensionality
    eff_rank_pre, pr_pre = compute_intrinsic_dimensionality(X_pre)
    eff_rank_post, pr_post = compute_intrinsic_dimensionality(X_post)

    # PCA Variance Spectrum
    n_comp = min(10, X_pre.shape[1], X_post.shape[1])
    pca_pre = PCA(n_components=n_comp).fit(X_pre)
    pca_post = PCA(n_components=n_comp).fit(X_post)

    var_pre = pca_pre.explained_variance_ratio_
    var_post = pca_post.explained_variance_ratio_

    print(f"Pre-Projection (Layer12+LN) : Effective Rank={eff_rank_pre:.2f}, Participation Ratio={pr_pre:.2f}, PC1={var_pre[0]*100:.2f}%")
    print(f"Post-Projection (Final L2)  : Effective Rank={eff_rank_post:.2f}, Participation Ratio={pr_post:.2f}, PC1={var_post[0]*100:.2f}%")

    fig, ax = plt.subplots(figsize=(8, 5))
    indices = np.arange(1, n_comp + 1)
    ax.plot(indices, var_pre * 100, "o-", color="seagreen", lw=2, label=f"Pre-Projection (r_eff={eff_rank_pre:.1f}, PR={pr_pre:.1f})")
    ax.plot(indices, var_post * 100, "s-", color="crimson", lw=2, label=f"Post-Projection (r_eff={eff_rank_post:.1f}, PR={pr_post:.1f})")
    ax.set_xlabel("Principal Component Index", fontsize=11)
    ax.set_ylabel("Explained Variance Ratio (%)", fontsize=11)
    ax.set_title("Representation Geometry Shift: PCA Variance Spectrum & Intrinsic Dimension", fontsize=11, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(fontsize=10)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "pca_spectrum_compression.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    report = {
        "pre_effective_rank": eff_rank_pre,
        "pre_participation_ratio": pr_pre,
        "pre_pc1_var": float(var_pre[0]),
        "post_effective_rank": eff_rank_post,
        "post_participation_ratio": pr_post,
        "post_pc1_var": float(var_post[0]),
    }

    rpt_path = os.path.join(output_dir, "pca_spectrum_report.json")
    with open(rpt_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


# ==============================================================================
# Stage 3: Text-side Projection Causal Ablation
# ==============================================================================

def analyze_projection_ablation(
    model: nn.Module,
    tokenizer: Any,
    pos_texts: List[str],
    neg_texts: List[str],
    output_dir: str,
    device: str = "cpu",
    batch_size: int = 256,
) -> Dict[str, Any]:
    """
    Causal Ablation: Replace W_proj with (1) Original W_proj, (2) Identity Matrix, (3) Random Orthogonal.
    Evaluated strictly in text representation space.
    """
    import pandas as pd

    print("\n" + "="*60)
    print("Stage 3: Projection Matrix Causal Ablation (Text Representation Space)")
    print("="*60)

    text_tower = getattr(model, 'text', model)
    text_projection = getattr(text_tower, 'text_projection', None)

    if text_projection is None:
        print("Model does not have a text_projection matrix. Skipping Ablation.")
        return {}

    if isinstance(text_projection, nn.Linear):
        W_orig = text_projection.weight.T.detach().cpu().numpy()
    else:
        W_orig = text_projection.detach().cpu().numpy()

    D_in, D_out = W_orig.shape

    # 1. Original W_proj
    steps_orig = extract_pipeline_step_features(model, tokenizer, pos_texts, device, batch_size, custom_projection=None)
    neg_orig = extract_pipeline_step_features(model, tokenizer, neg_texts, device, batch_size, custom_projection=None)

    # 2. Identity Matrix
    steps_ident = extract_pipeline_step_features(model, tokenizer, pos_texts, device, batch_size, custom_projection="identity")
    neg_ident = extract_pipeline_step_features(model, tokenizer, neg_texts, device, batch_size, custom_projection="identity")

    # 3. Random Orthogonal Matrix
    np.random.seed(42)
    Q, _ = np.linalg.qr(np.random.randn(D_in, D_out))
    steps_orth = extract_pipeline_step_features(model, tokenizer, pos_texts, device, batch_size, custom_projection=Q)
    neg_orth = extract_pipeline_step_features(model, tokenizer, neg_texts, device, batch_size, custom_projection=Q)

    def get_sims(pos_map, neg_map):
        pos_f = pos_map["Step4_Final_L2Norm"]
        neg_f = neg_map["Step4_Final_L2Norm"]
        pos_n = pos_f / (np.linalg.norm(pos_f, axis=1, keepdims=True) + 1e-8)
        neg_n = neg_f / (np.linalg.norm(neg_f, axis=1, keepdims=True) + 1e-8)
        return np.sum(pos_n * neg_n, axis=1)

    sims_orig = get_sims(steps_orig, neg_orig)
    sims_ident = get_sims(steps_ident, neg_ident)
    sims_orth = get_sims(steps_orth, neg_orth)

    ablation_results = [
        {"projection_condition": "Original Trained W_proj", "mean_cosine_sim": float(np.mean(sims_orig)), "std_cosine_sim": float(np.std(sims_orig))},
        {"projection_condition": "Identity Matrix (No Proj)", "mean_cosine_sim": float(np.mean(sims_ident)), "std_cosine_sim": float(np.std(sims_ident))},
        {"projection_condition": "Random Orthogonal Matrix Q", "mean_cosine_sim": float(np.mean(sims_orth)), "std_cosine_sim": float(np.std(sims_orth))},
    ]

    for r in ablation_results:
        print(f"  [{r['projection_condition']}] Mean Cosine Sim: {r['mean_cosine_sim']:.4f}")

    df_ablation = pd.DataFrame(ablation_results)
    csv_path = os.path.join(output_dir, "projection_causal_ablation.csv")
    df_ablation.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(df_ablation["projection_condition"], df_ablation["mean_cosine_sim"], color=["crimson", "seagreen", "dodgerblue"], alpha=0.85, edgecolor="black")
    ax.set_ylabel("Final Cosine Similarity (Lower = Better Negation Separation)", fontsize=10)
    ax.set_title("Causal Ablation: How Projection Matrix Choice Affects Text Similarity", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", ls="--", alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "projection_causal_ablation.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    return {"csv_path": csv_path, "plot_path": plot_path, "ablation_results": ablation_results}


# ==============================================================================
# Stage 2: Image-Text Retrieval Metrics & Ranking Flip Rate
# ==============================================================================

def analyze_image_text_retrieval_metrics(
    model: nn.Module,
    tokenizer: Any,
    preprocess: Any,
    pair_metadata: List[dict],
    pos_texts: List[str],
    neg_texts: List[str],
    image_root: str,
    output_dir: str,
    device: str = "cpu",
    batch_size: int = 64,
):
    """
    Compute Accuracy, MRR, Ranking Flip Rate, and Pearson correlation.
    """
    import pandas as pd
    from PIL import Image

    model.eval()
    results = []
    skipped = 0

    image_pairs = {}
    for i, meta in enumerate(pair_metadata):
        img_path = os.path.join(image_root, meta["image_path"])
        if img_path not in image_pairs:
            image_pairs[img_path] = []
        image_pairs[img_path].append(i)

    print(f"Processing {len(image_pairs)} unique images for Retrieval Metrics...")
    for img_path, indices in image_pairs.items():
        if not os.path.exists(img_path):
            skipped += len(indices)
            continue

        try:
            image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        except Exception:
            skipped += len(indices)
            continue

        with torch.no_grad():
            image_emb = model.encode_image(image, normalize=True).float().cpu()

        for idx in indices:
            pos_tok = tokenizer([pos_texts[idx]]).to(device)
            neg_tok = tokenizer([neg_texts[idx]]).to(device)

            with torch.no_grad():
                pos_emb = model.encode_text(pos_tok, normalize=True).float().cpu()
                neg_emb = model.encode_text(neg_tok, normalize=True).float().cpu()

            sim_pos = float(F.cosine_similarity(image_emb, pos_emb).item())
            sim_neg = float(F.cosine_similarity(image_emb, neg_emb).item())

            results.append({
                "image_path": pair_metadata[idx]["image_path"],
                "object_name": pair_metadata[idx].get("object_name", ""),
                "object_in_image": pair_metadata[idx].get("object_in_image", ""),
                "positive_caption": pos_texts[idx],
                "negative_caption": neg_texts[idx],
                "sim_image_pos": sim_pos,
                "sim_image_neg": sim_neg,
                "sim_diff": sim_pos - sim_neg,
            })

    if len(results) == 0:
        print("No valid images found. Skipping Retrieval Metrics.")
        return

    res_df = pd.DataFrame(results)
    res_path = os.path.join(output_dir, "image_text_similarity.csv")
    res_df.to_csv(res_path, index=False)

    sim_pos_arr = res_df["sim_image_pos"].values
    sim_neg_arr = res_df["sim_image_neg"].values
    sim_diff_arr = res_df["sim_diff"].values

    pearson_r = float(np.corrcoef(sim_pos_arr, sim_neg_arr)[0, 1])

    in_img_mask = res_df["object_in_image"] == True
    if np.sum(in_img_mask) > 0:
        flip_rate_in_img = float(np.mean(sim_neg_arr[in_img_mask] > sim_pos_arr[in_img_mask])) * 100
        accuracy_in_img = float(np.mean(sim_pos_arr[in_img_mask] > sim_neg_arr[in_img_mask])) * 100
    else:
        flip_rate_in_img = 0.0
        accuracy_in_img = 0.0

    retrieval_summary = {
        "total_pairs_evaluated": len(results),
        "pearson_r": pearson_r,
        "mean_sim_pos": float(np.mean(sim_pos_arr)),
        "mean_sim_neg": float(np.mean(sim_neg_arr)),
        "mean_sim_diff": float(np.mean(sim_diff_arr)),
        "binary_mcq_accuracy_pct": accuracy_in_img,
        "ranking_flip_rate_pct": flip_rate_in_img,
    }

    sum_path = os.path.join(output_dir, "retrieval_metrics_summary.json")
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(retrieval_summary, f, indent=2)

    print("\n=== Retrieval Metrics Summary ===")
    print(f"  Pearson r                 : {pearson_r:.4f}")
    print(f"  Binary MCQ Accuracy       : {accuracy_in_img:.1f}%")
    print(f"  Ranking Flip Rate         : {flip_rate_in_img:.1f}%")


# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    import pandas as pd

    parser = argparse.ArgumentParser(description="CLIP Negation Analysis Refined 3rd Edition")
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--target_token", type=str, default="eot", choices=["eot", "mean", "all"])
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="logs/pipeline_breakdown/openai_vit_b32")
    parser.add_argument("--max_samples", type=int, default=60000)
    parser.add_argument("--image_root", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    pos_texts = []
    neg_texts = []
    pair_metadata = []

    if args.csv_path and os.path.exists(args.csv_path):
        df = pd.read_csv(args.csv_path).head(args.max_samples)
        pos_texts = df["positive_caption"].astype(str).tolist()
        neg_texts = df["negative_caption"].astype(str).tolist()
        for _, row in df.iterrows():
            meta = {
                "image_path": str(row.get("image_path", "")),
                "object_name": str(row.get("object_name", "")),
                "object_in_image": row.get("object_in_image", None)
            }
            if isinstance(meta["object_in_image"], str):
                meta["object_in_image"] = meta["object_in_image"].strip().lower() == "true"
            pair_metadata.append(meta)

    # Load model
    print(f"Loading model {args.model} ({args.pretrained})...")
    model, preprocess, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device)

    # Stage 1-A. Pipeline 5-Step Breakdown Analysis
    analyze_pipeline_breakdown(model, tokenizer, pos_texts, neg_texts, args.output_dir, device, args.batch_size)

    # Stage 1-B. Direction Preservation Analysis (Distance Ratio)
    analyze_direction_preservation(model, tokenizer, pos_texts, neg_texts, args.output_dir, device, args.batch_size)

    # Stage 1-C. Linear Probe Analysis (Linear Separability Pre vs Post)
    analyze_linear_probe(model, tokenizer, pos_texts, neg_texts, args.output_dir, device, args.batch_size)

    # Stage 1-D. Intrinsic Dimensionality & PCA Spectrum (Effective Rank & Participation Ratio)
    analyze_pca_spectrum_compression(model, tokenizer, pos_texts, neg_texts, args.output_dir, device, args.batch_size)

    # Stage 3. Projection Matrix Causal Ablation (Text Space)
    analyze_projection_ablation(model, tokenizer, pos_texts, neg_texts, args.output_dir, device, args.batch_size)

    # Stage 2. Retrieval Metrics (if image_root provided)
    if args.image_root:
        analyze_image_text_retrieval_metrics(model, tokenizer, preprocess, pair_metadata, pos_texts, neg_texts, args.image_root, args.output_dir, device, args.batch_size)

    print(f"\n✅ Refined 3rd Edition Pipeline Analysis Complete! Results saved in: {args.output_dir}")
