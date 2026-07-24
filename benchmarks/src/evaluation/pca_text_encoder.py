"""
CLIP Text Encoder Layer-wise PCA & Cosine Similarity Analysis Module.

Supports two modes:
  1. Unpaired mode (MCQ CSV): Separate positive/negative text lists → PCA only
  2. Paired mode (Paired CSV): Matched positive/negative pairs → PCA + per-pair cosine similarity

Optionally, if image paths are available and images exist, computes image-text
cosine similarity correlation analysis (Experiment 3).
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any

import open_clip


# ==============================================================================
# Core: Layer-wise feature extraction
# ==============================================================================

def extract_layer_representations(
    model,
    tokenizer,
    texts: List[str],
    device: str = "cpu",
    target_token: str = "eot",
    batch_size: int = 256,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Extract layer-wise representations and final projected embeddings.

    Returns:
        layer_features: List[np.ndarray] of shape (N, D) per layer.
        final_embeddings: np.ndarray of shape (N, embed_dim) — projected, normalized.
    """
    model.eval()
    all_tokens = tokenizer(texts).to(device)

    all_layer_features = None
    all_final_embeddings = []

    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch_tokens = all_tokens[start:end]

        with torch.no_grad():
            final_emb, hidden_states = model.encode_text(
                batch_tokens, normalize=True, return_all_layers=True
            )

        all_final_embeddings.append(final_emb.float().cpu())

        if all_layer_features is None:
            all_layer_features = [[] for _ in range(len(hidden_states))]

        for l_idx, hs in enumerate(hidden_states):
            hs_cpu = hs.float().cpu()
            if target_token == "eot":
                eot_indices = batch_tokens.argmax(dim=-1).cpu()
                batch_indices = torch.arange(hs_cpu.shape[0])
                feat = hs_cpu[batch_indices, eot_indices]
            elif target_token == "mean":
                feat = hs_cpu.mean(dim=1)
            elif target_token == "all":
                feat = hs_cpu.reshape(-1, hs_cpu.shape[-1])
            else:
                raise ValueError(f"Unknown target_token: {target_token}")
            all_layer_features[l_idx].append(feat)

    layer_features = [torch.cat(feats, dim=0).numpy() for feats in all_layer_features]
    final_embeddings = torch.cat(all_final_embeddings, dim=0).numpy()

    return layer_features, final_embeddings


# ==============================================================================
# Experiment 1 & 2-A: PCA analysis (layer-wise)
# ==============================================================================

def analyze_pca(
    pos_layer_feats: List[np.ndarray],
    neg_layer_feats: List[np.ndarray],
    n_pos: int,
    n_neg: int,
    output_dir: str,
    target_token: str,
    model_name: str,
    pretrained: str,
) -> List[dict]:
    """PCA analysis and grid plot."""
    num_layers = len(pos_layer_feats)

    cols = min(4, num_layers)
    rows = (num_layers + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    if num_layers == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    report = []
    report.append("=== CLIP Text Encoder Layer-wise PCA Analysis Report ===")
    report.append(f"Model: {model_name} ({pretrained})")
    report.append(f"Target Token: {target_token}")
    report.append(f"Positive: {n_pos}, Negative: {n_neg}")
    report.append(f"Layers: {num_layers} (Layer 0 = Embedding)\n")

    pca_data = []

    for l_idx in range(num_layers):
        pos_f = pos_layer_feats[l_idx]
        neg_f = neg_layer_feats[l_idx]
        combined = np.vstack([pos_f, neg_f])

        pca = PCA(n_components=min(combined.shape[0], combined.shape[1], 2))
        combined_pca = pca.fit_transform(combined)
        pos_pca = combined_pca[:n_pos]
        neg_pca = combined_pca[n_pos:]

        var_ratio = pca.explained_variance_ratio_
        total_var = float(np.sum(var_ratio[:2])) if len(var_ratio) >= 2 else float(np.sum(var_ratio))

        pos_mean_orig = pos_f.mean(axis=0)
        neg_mean_orig = neg_f.mean(axis=0)
        cdist = float(np.linalg.norm(pos_mean_orig - neg_mean_orig))

        pos_mean_pca = pos_pca.mean(axis=0)
        neg_mean_pca = neg_pca.mean(axis=0)

        layer_name = "Embedding" if l_idx == 0 else f"Layer {l_idx}"
        pca_data.append({
            "layer": layer_name, "explained_var_2d": total_var,
            "pc1_var": float(var_ratio[0]), "pc2_var": float(var_ratio[1]) if len(var_ratio) > 1 else 0,
            "centroid_dist": cdist
        })
        report.append(f"[{layer_name}] Var: {total_var*100:.1f}% | Centroid Dist: {cdist:.4f}")

        ax = axes[l_idx]
        ax.scatter(pos_pca[:, 0], pos_pca[:, 1], c="dodgerblue", label="Positive",
                   alpha=0.7, edgecolors="k", linewidth=0.5, s=40)
        ax.scatter(neg_pca[:, 0], neg_pca[:, 1], c="crimson", label="Negative",
                   alpha=0.7, edgecolors="k", linewidth=0.5, marker="^", s=40)
        ax.scatter(*pos_mean_pca, c="navy", s=120, marker="X", label="Pos Centroid", edgecolors="w")
        ax.scatter(*neg_mean_pca, c="darkred", s=120, marker="X", label="Neg Centroid", edgecolors="w")
        ax.set_title(f"{layer_name}\n(Var: {total_var*100:.1f}%)", fontsize=11, fontweight="bold")
        ax.set_xlabel("PC 1", fontsize=9)
        ax.set_ylabel("PC 2", fontsize=9)
        ax.grid(True, ls="--", alpha=0.5)
        if l_idx == 0:
            ax.legend(fontsize=8, loc="best")

    for l_idx in range(num_layers, len(axes)):
        fig.delaxes(axes[l_idx])

    plt.tight_layout()
    path = os.path.join(output_dir, f"pca_grid_{target_token}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    rpt_path = os.path.join(output_dir, f"pca_report_{target_token}.txt")
    with open(rpt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"Saved: {rpt_path}")
    print("\n".join(report))

    return pca_data


# ==============================================================================
# Experiment 2-B,C: Paired cosine similarity analysis
# ==============================================================================

def analyze_paired_cosine_similarity(
    pos_layer_feats: List[np.ndarray],
    neg_layer_feats: List[np.ndarray],
    pos_final_emb: np.ndarray,
    neg_final_emb: np.ndarray,
    pair_metadata: Optional[List[dict]],
    output_dir: str,
    target_token: str,
):
    """
    Compute per-pair cosine similarity at each layer and at the final embedding.
    Requires pos and neg to be aligned (same length, index i is a matched pair).
    """
    import pandas as pd

    n_pairs = pos_final_emb.shape[0]
    assert neg_final_emb.shape[0] == n_pairs, "Positive and negative must be same length for paired analysis"
    num_layers = len(pos_layer_feats)

    # --- 2-B: Layer-wise cosine similarity ---
    layer_sim_matrix = np.zeros((n_pairs, num_layers))  # (N, L)
    layer_stats = []

    for l_idx in range(num_layers):
        pf = pos_layer_feats[l_idx]  # (N, D)
        nf = neg_layer_feats[l_idx]
        # Cosine similarity per pair
        pf_norm = pf / (np.linalg.norm(pf, axis=1, keepdims=True) + 1e-8)
        nf_norm = nf / (np.linalg.norm(nf, axis=1, keepdims=True) + 1e-8)
        sims = np.sum(pf_norm * nf_norm, axis=1)  # (N,)
        layer_sim_matrix[:, l_idx] = sims

        layer_name = "Embedding" if l_idx == 0 else f"Layer {l_idx}"
        layer_stats.append({
            "layer": layer_name,
            "mean": float(np.mean(sims)),
            "median": float(np.median(sims)),
            "std": float(np.std(sims)),
            "min": float(np.min(sims)),
            "max": float(np.max(sims)),
        })

    # Final embedding cosine similarity
    pf_norm = pos_final_emb / (np.linalg.norm(pos_final_emb, axis=1, keepdims=True) + 1e-8)
    nf_norm = neg_final_emb / (np.linalg.norm(neg_final_emb, axis=1, keepdims=True) + 1e-8)
    final_sims = np.sum(pf_norm * nf_norm, axis=1)
    layer_stats.append({
        "layer": "Final (projected)",
        "mean": float(np.mean(final_sims)),
        "median": float(np.median(final_sims)),
        "std": float(np.std(final_sims)),
        "min": float(np.min(final_sims)),
        "max": float(np.max(final_sims)),
    })

    # Save layer stats CSV
    stats_df = pd.DataFrame(layer_stats)
    stats_path = os.path.join(output_dir, f"cosine_similarity_by_layer_{target_token}.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved: {stats_path}")

    # Save per-pair similarity CSV
    pair_cols = {}
    for l_idx in range(num_layers):
        layer_name = "Embedding" if l_idx == 0 else f"Layer_{l_idx}"
        pair_cols[f"sim_{layer_name}"] = layer_sim_matrix[:, l_idx]
    pair_cols["sim_Final_projected"] = final_sims

    if pair_metadata:
        for key in pair_metadata[0].keys():
            pair_cols[key] = [m[key] for m in pair_metadata]

    pairs_df = pd.DataFrame(pair_cols)
    pairs_path = os.path.join(output_dir, f"cosine_similarity_pairs_{target_token}.csv")
    pairs_df.to_csv(pairs_path, index=False)
    print(f"Saved: {pairs_path}")

    # --- Plot 1: Layer-wise cosine similarity distribution (box plot) ---
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    plot_data = []
    for l_idx in range(num_layers):
        layer_name = "Emb" if l_idx == 0 else f"L{l_idx}"
        for s in layer_sim_matrix[:, l_idx]:
            plot_data.append({"Layer": layer_name, "Cosine Similarity": s})
    for s in final_sims:
        plot_data.append({"Layer": "Final", "Cosine Similarity": s})
    plot_df = pd.DataFrame(plot_data)
    sns.boxplot(data=plot_df, x="Layer", y="Cosine Similarity", ax=ax, palette="coolwarm", showfliers=False)
    sns.stripplot(data=plot_df, x="Layer", y="Cosine Similarity", ax=ax,
                  color="black", alpha=0.15, size=2, jitter=True)
    ax.set_title("Positive↔Negative Pair Cosine Similarity by Layer", fontsize=13, fontweight="bold")
    ax.set_ylabel("Cosine Similarity", fontsize=11)
    ax.axhline(y=1.0, color="gray", ls="--", alpha=0.5, label="Perfect similarity")
    ax.grid(True, axis="y", ls="--", alpha=0.3)
    plt.tight_layout()
    dist_path = os.path.join(output_dir, f"cosine_similarity_distribution_{target_token}.png")
    plt.savefig(dist_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {dist_path}")

    # --- Plot 2: Final embedding cosine similarity histogram ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.hist(final_sims, bins=40, color="steelblue", edgecolor="black", alpha=0.8)
    ax.axvline(x=np.mean(final_sims), color="red", ls="--", lw=2,
               label=f"Mean: {np.mean(final_sims):.3f}")
    ax.axvline(x=np.median(final_sims), color="orange", ls="--", lw=2,
               label=f"Median: {np.median(final_sims):.3f}")
    ax.set_title("Final Embedding: Positive↔Negative Cosine Similarity", fontsize=13, fontweight="bold")
    ax.set_xlabel("Cosine Similarity", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", ls="--", alpha=0.3)
    plt.tight_layout()
    hist_path = os.path.join(output_dir, f"cosine_similarity_final_histogram_{target_token}.png")
    plt.savefig(hist_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {hist_path}")

    # --- Plot 3: Group comparison if object_in_image info available ---
    if pair_metadata and "object_in_image" in pair_metadata[0]:
        in_img = np.array([m["object_in_image"] for m in pair_metadata])
        sims_in = final_sims[in_img == True]
        sims_out = final_sims[in_img == False]

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.hist(sims_in, bins=30, alpha=0.6, color="dodgerblue", edgecolor="black",
                label=f"Object IN image (n={len(sims_in)}, mean={np.mean(sims_in):.3f})")
        ax.hist(sims_out, bins=30, alpha=0.6, color="crimson", edgecolor="black",
                label=f"Object NOT in image (n={len(sims_out)}, mean={np.mean(sims_out):.3f})")
        ax.set_title("Final Cosine Sim: Object In Image vs Not", fontsize=13, fontweight="bold")
        ax.set_xlabel("Cosine Similarity", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, axis="y", ls="--", alpha=0.3)
        plt.tight_layout()
        grp_path = os.path.join(output_dir, f"cosine_similarity_by_group_{target_token}.png")
        plt.savefig(grp_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {grp_path}")

    # Print summary
    print("\n=== Cosine Similarity Summary ===")
    for s in layer_stats:
        print(f"  [{s['layer']}] mean={s['mean']:.4f}  median={s['median']:.4f}  std={s['std']:.4f}")


# ==============================================================================
# Experiment 3: Image-Text cosine similarity correlation (requires images)
# ==============================================================================

def analyze_image_text_correlation(
    model,
    tokenizer,
    preprocess,
    pair_metadata: List[dict],
    pos_texts: List[str],
    neg_texts: List[str],
    image_root: str,
    output_dir: str,
    device: str = "cpu",
    batch_size: int = 64,
):
    """
    Compute cos_sim(image, positive_caption) vs cos_sim(image, negative_caption)
    and analyze their correlation.
    """
    import pandas as pd
    from PIL import Image
    from scipy import stats

    model.eval()
    results = []
    skipped = 0

    # Group by unique image paths to avoid redundant image encoding
    image_pairs = {}
    for i, meta in enumerate(pair_metadata):
        img_path = os.path.join(image_root, meta["image_path"])
        if img_path not in image_pairs:
            image_pairs[img_path] = []
        image_pairs[img_path].append(i)

    print(f"Processing {len(image_pairs)} unique images...")
    for img_path, indices in image_pairs.items():
        if not os.path.exists(img_path):
            skipped += len(indices)
            continue

        try:
            image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        except Exception as e:
            print(f"  Error loading {img_path}: {e}")
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

    if skipped > 0:
        print(f"  Skipped {skipped} pairs (missing images)")

    if len(results) == 0:
        print("No valid image-text pairs found. Skipping Experiment 3.")
        return

    res_df = pd.DataFrame(results)
    res_path = os.path.join(output_dir, "image_text_similarity.csv")
    res_df.to_csv(res_path, index=False)
    print(f"Saved: {res_path}")

    sim_pos_arr = res_df["sim_image_pos"].values
    sim_neg_arr = res_df["sim_image_neg"].values

    pearson_r, pearson_p = stats.pearsonr(sim_pos_arr, sim_neg_arr)
    spearman_r, spearman_p = stats.spearmanr(sim_pos_arr, sim_neg_arr)

    # Scatter plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    if "object_in_image" in res_df.columns:
        in_mask = res_df["object_in_image"] == True
        ax.scatter(sim_pos_arr[in_mask], sim_neg_arr[in_mask],
                   c="dodgerblue", alpha=0.6, s=30, label="Object IN image", edgecolors="k", linewidth=0.3)
        ax.scatter(sim_pos_arr[~in_mask], sim_neg_arr[~in_mask],
                   c="crimson", alpha=0.6, s=30, label="Object NOT in image", marker="^", edgecolors="k", linewidth=0.3)
    else:
        ax.scatter(sim_pos_arr, sim_neg_arr, c="steelblue", alpha=0.6, s=30, edgecolors="k", linewidth=0.3)

    # Diagonal
    lims = [min(sim_pos_arr.min(), sim_neg_arr.min()) - 0.05,
            max(sim_pos_arr.max(), sim_neg_arr.max()) + 0.05]
    ax.plot(lims, lims, "k--", alpha=0.3, label="y=x (perfect correlation)")

    ax.set_xlabel('cos_sim(image, "There is A")', fontsize=12)
    ax.set_ylabel('cos_sim(image, "There is no A")', fontsize=12)
    ax.set_title(f"Image-Text Similarity: Positive vs Negative\n"
                 f"Pearson r={pearson_r:.3f} (p={pearson_p:.1e}), "
                 f"Spearman ρ={spearman_r:.3f} (p={spearman_p:.1e})",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, ls="--", alpha=0.3)
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, "image_text_correlation.png")
    plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {scatter_path}")

    # Summary
    print(f"\n=== Image-Text Correlation Summary ===")
    print(f"  Valid pairs     : {len(results)}")
    print(f"  Pearson  r      : {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"  Spearman ρ      : {spearman_r:.4f} (p={spearman_p:.2e})")
    print(f"  Mean sim(pos)   : {np.mean(sim_pos_arr):.4f}")
    print(f"  Mean sim(neg)   : {np.mean(sim_neg_arr):.4f}")
    print(f"  Mean diff       : {np.mean(sim_pos_arr - sim_neg_arr):.4f}")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    import pandas as pd

    parser = argparse.ArgumentParser(description="CLIP Negation Analysis: PCA + Cosine Similarity")
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--target_token", type=str, default="eot", choices=["eot", "mean", "all"])
    parser.add_argument("--csv_path", type=str, default=None, help="Path to CSV (MCQ or Paired format)")
    parser.add_argument("--output_dir", type=str, default="logs/pca/default")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--image_root", type=str, default="", help="Root path for images (for Experiment 3)")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load CSV ──
    pos_texts = []
    neg_texts = []
    pair_metadata = None
    is_paired = False

    if args.csv_path and os.path.exists(args.csv_path):
        df = pd.read_csv(args.csv_path)
        print(f"Loaded CSV: {args.csv_path} ({len(df)} rows)")

        # Auto-detect format
        if "positive_caption" in df.columns and "negative_caption" in df.columns:
            # ── Paired format ──
            is_paired = True
            df = df.head(args.max_samples)
            pos_texts = df["positive_caption"].astype(str).tolist()
            neg_texts = df["negative_caption"].astype(str).tolist()
            pair_metadata = []
            for _, row in df.iterrows():
                meta = {"image_path": str(row.get("image_path", "")),
                        "object_name": str(row.get("object_name", "")),
                        "object_in_image": row.get("object_in_image", None)}
                # Handle string booleans from CSV
                if isinstance(meta["object_in_image"], str):
                    meta["object_in_image"] = meta["object_in_image"].strip().lower() == "true"
                pair_metadata.append(meta)
            print(f"Paired mode: {len(pos_texts)} matched pairs")
        else:
            # ── MCQ format ──
            for _, row in df.iterrows():
                correct_idx = int(row.get("correct_answer", 0))
                pos_col = f"caption_{correct_idx}"
                if pos_col in row and pd.notna(row[pos_col]):
                    pos_texts.append(str(row[pos_col]))
                for c_i in range(4):
                    if c_i != correct_idx:
                        c_col = f"caption_{c_i}"
                        if c_col in row and pd.notna(row[c_col]):
                            neg_texts.append(str(row[c_col]))
                if len(pos_texts) >= args.max_samples:
                    break
            print(f"MCQ mode: {len(pos_texts)} pos, {len(neg_texts)} neg")
    else:
        print("No CSV. Using demo texts.")
        pos_texts = [
            "A photo of a dog in the park", "A person sitting on a red sofa",
            "A bright sunny day at the beach", "A woman holding a cell phone",
        ]
        neg_texts = [
            "A photo of a park without any dog", "A person sitting on a sofa, but no red sofa visible",
            "A beach scene with no bright sun", "A woman standing with no cell phone in sight",
        ]

    # ── Load model ──
    print(f"Loading model {args.model} ({args.pretrained})...")
    model, preprocess, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device)

    # ── Extract features ──
    print(f"Extracting positive text features ({len(pos_texts)} texts)...")
    pos_layer_feats, pos_final_emb = extract_layer_representations(
        model, tokenizer, pos_texts, device, args.target_token, args.batch_size)

    print(f"Extracting negative text features ({len(neg_texts)} texts)...")
    neg_layer_feats, neg_final_emb = extract_layer_representations(
        model, tokenizer, neg_texts, device, args.target_token, args.batch_size)

    # ── Experiment 1: PCA ──
    print("\n" + "="*60)
    print("Experiment 1: Layer-wise PCA Analysis")
    print("="*60)
    pca_data = analyze_pca(
        pos_layer_feats, neg_layer_feats,
        len(pos_texts), len(neg_texts),
        args.output_dir, args.target_token, args.model, args.pretrained)

    # ── Experiment 2: Paired Cosine Similarity ──
    if is_paired:
        print("\n" + "="*60)
        print("Experiment 2: Paired Cosine Similarity Analysis")
        print("="*60)
        analyze_paired_cosine_similarity(
            pos_layer_feats, neg_layer_feats,
            pos_final_emb, neg_final_emb,
            pair_metadata, args.output_dir, args.target_token)

    # ── Experiment 3: Image-Text Correlation (if image_root provided) ──
    if args.image_root and is_paired:
        print("\n" + "="*60)
        print("Experiment 3: Image-Text Cosine Similarity Correlation")
        print("="*60)
        analyze_image_text_correlation(
            model, tokenizer, preprocess, pair_metadata,
            pos_texts, neg_texts,
            args.image_root, args.output_dir, device, args.batch_size)
    elif args.image_root and not is_paired:
        print("\nSkipping Experiment 3: requires paired CSV format.")

    print("\n✅ All experiments complete. Results in:", args.output_dir)
