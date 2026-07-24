"""
CLIP Text Encoder Layer-wise PCA Analysis and Visualization Module.

This script extracts hidden representations from all layers of CLIP text encoder,
applies PCA (Principal Component Analysis), and visualizes positive vs. negative text embeddings
on unified plots per layer with detailed variance & distance analysis.
"""

import os
import argparse
import numpy as np
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any

import open_clip


def extract_layer_representations(
    model,
    tokenizer,
    texts: List[str],
    device: str = "cpu",
    target_token: str = "eot"  # Options: 'eot' (default), 'mean', 'all'
) -> List[np.ndarray]:
    """
    Extract layer-wise representations for a list of input texts.

    Args:
        model: CLIP model instance.
        tokenizer: OpenCLIP tokenizer.
        texts: List of text strings.
        device: 'cuda' or 'cpu'.
        target_token: Token aggregation strategy ('eot', 'mean', 'all').

    Returns:
        List of numpy arrays, where index i corresponds to Layer i hidden states of shape (N, Dim).
    """
    model.eval()
    tokens = tokenizer(texts).to(device)

    with torch.no_grad():
        _, hidden_states = model.encode_text(tokens, return_all_layers=True)

    layer_features = []
    # hidden_states: List of Tensors, each shape [batch_size, seq_len, width]
    for layer_idx, hs in enumerate(hidden_states):
        # hs shape: [B, L, D]
        hs_cpu = hs.float().cpu()
        
        if target_token == "eot":
            # EOT token is the token with maximum index in sequence (argmax of token ids)
            eot_indices = tokens.argmax(dim=-1).cpu()
            batch_indices = torch.arange(hs_cpu.shape[0])
            feat = hs_cpu[batch_indices, eot_indices].numpy()  # [B, D]
        elif target_token == "mean":
            feat = hs_cpu.mean(dim=1).numpy()  # [B, D]
        elif target_token == "all":
            feat = hs_cpu.reshape(-1, hs_cpu.shape[-1]).numpy()  # [B * L, D]
        else:
            raise ValueError(f"Unknown target_token strategy: {target_token}")

        layer_features.append(feat)

    return layer_features


def analyze_and_plot_pca(
    pos_texts: List[str],
    neg_texts: List[str],
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    target_token: str = "eot",
    output_dir: str = "pca_results",
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Perform layer-wise PCA on positive vs negative texts, plot single-canvas comparative figures,
    and generate analytical summary.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading model {model_name} ({pretrained})...")
    model, _, tokenizer = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device)

    # Extract layer representations
    print(f"Extracting hidden states (Target token strategy: '{target_token}')...")
    pos_layer_feats = extract_layer_representations(model, tokenizer, pos_texts, device, target_token)
    neg_layer_feats = extract_layer_representations(model, tokenizer, neg_texts, device, target_token)

    num_layers = len(pos_layer_feats)  # Layer 0 (Embedding) + L transformer layers
    n_pos = len(pos_texts)
    n_neg = len(neg_texts)

    # Setup matplotlib grid
    cols = min(4, num_layers)
    rows = (num_layers + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    if num_layers == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    analysis_report = []
    analysis_report.append(f"=== CLIP Text Encoder Layer-wise PCA Analysis Report ===")
    analysis_report.append(f"Model: {model_name} ({pretrained})")
    analysis_report.append(f"Target Token Strategy: {target_token}")
    analysis_report.append(f"Positive samples: {n_pos}, Negative samples: {n_neg}")
    analysis_report.append(f"Total Layers analyzed: {num_layers} (Layer 0 = Input Embedding)\n")

    pca_data_per_layer = []

    for l_idx in range(num_layers):
        pos_f = pos_layer_feats[l_idx]
        neg_f = neg_layer_feats[l_idx]

        combined = np.vstack([pos_f, neg_f])
        
        # Fit PCA on combined features
        pca = PCA(n_components=min(combined.shape[0], combined.shape[1], 2))
        combined_pca = pca.fit_transform(combined)

        pos_pca = combined_pca[:n_pos]
        neg_pca = combined_pca[n_pos:]

        var_ratio = pca.explained_variance_ratio_
        total_var_2d = float(np.sum(var_ratio[:2])) if len(var_ratio) >= 2 else float(np.sum(var_ratio))

        # Calculate centroid distance between Positive and Negative groups in PCA space & original space
        pos_mean_orig = pos_f.mean(axis=0)
        neg_mean_orig = neg_f.mean(axis=0)
        centroid_dist_orig = float(np.linalg.norm(pos_mean_orig - neg_mean_orig))

        pos_mean_pca = pos_pca.mean(axis=0)
        neg_mean_pca = neg_pca.mean(axis=0)
        centroid_dist_pca = float(np.linalg.norm(pos_mean_pca - neg_mean_pca))

        layer_name = "Embedding" if l_idx == 0 else f"Layer {l_idx}"

        layer_info = {
            "layer_index": l_idx,
            "layer_name": layer_name,
            "explained_variance_ratio": var_ratio.tolist(),
            "total_explained_variance_2d": total_var_2d,
            "centroid_distance_orig": centroid_dist_orig,
            "centroid_distance_pca": centroid_dist_pca
        }
        pca_data_per_layer.append(layer_info)

        report_str = (f"[{layer_name}] 2D Explained Variance: {total_var_2d*100:.2f}% "
                      f"(PC1: {var_ratio[0]*100:.1f}%, PC2: {var_ratio[1]*100:.1f}%) | "
                      f"Group Centroid Dist (Orig Dim): {centroid_dist_orig:.4f}")
        analysis_report.append(report_str)

        # Plotting
        ax = axes[l_idx]
        ax.scatter(pos_pca[:, 0], pos_pca[:, 1], c="dodgerblue", label="Positive", alpha=0.75, edgecolors="k", linewidth=0.5, s=40)
        ax.scatter(neg_pca[:, 0], neg_pca[:, 1], c="crimson", label="Negative", alpha=0.75, edgecolors="k", linewidth=0.5, marker="^", s=40)

        # Centroids
        ax.scatter(pos_mean_pca[0], pos_mean_pca[1], c="navy", s=120, marker="X", label="Pos Centroid", edgecolors="w")
        ax.scatter(neg_mean_pca[0], neg_mean_pca[1], c="darkred", s=120, marker="X", label="Neg Centroid", edgecolors="w")

        ax.set_title(f"{layer_name}\n(Var: {total_var_2d*100:.1f}%)", fontsize=11, fontweight="bold")
        ax.set_xlabel("PC 1", fontsize=9)
        ax.set_ylabel("PC 2", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)

        if l_idx == 0:
            ax.legend(fontsize=8, loc="best")

    # Hide extra unused subplots
    for l_idx in range(num_layers, len(axes)):
        fig.delaxes(axes[l_idx])

    plt.tight_layout()
    plot_filename = os.path.join(output_dir, f"clip_layer_pca_{target_token}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved PCA plot grid to: {plot_filename}")

    # Write report to file
    report_filename = os.path.join(output_dir, f"pca_analysis_report_{target_token}.txt")
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(analysis_report))
    print(f"Saved Analysis report to: {report_filename}")

    # Print summary to stdout
    print("\n" + "\n".join(analysis_report))

    return {
        "plot_path": plot_filename,
        "report_path": report_filename,
        "layer_data": pca_data_per_layer
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP Layer-wise PCA Analysis")
    parser.add_argument("--model", type=str, default="ViT-B-32", help="CLIP model architecture")
    parser.add_argument("--pretrained", type=str, default="openai", help="Pretrained weights name/path")
    parser.add_argument("--target_token", type=str, default="eot", choices=["eot", "mean", "all"], help="Token representation strategy")
    parser.add_argument("--output_dir", type=str, default="pca_results", help="Directory to save plots and reports")
    args = parser.parse_args()

    # Sample demo texts (Positive vs Negative pairs)
    sample_positives = [
        "A photo of a dog in the park",
        "A person sitting on a red sofa",
        "A bright sunny day at the beach",
        "A woman holding a cell phone",
        "An apple on a wooden table",
        "A car driving on a highway",
        "A cup of hot coffee on the desk",
        "A cat sleeping peacefully on a rug"
    ]

    sample_negatives = [
        "A photo of a park without any dog",
        "A person sitting on a sofa, but no red sofa visible",
        "A beach scene with no bright sun",
        "A woman standing with no cell phone in sight",
        "A wooden table with no apple on it",
        "A highway with no car driving on it",
        "A desk with no cup of hot coffee",
        "A rug with no cat sleeping on it"
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    analyze_and_plot_pca(
        pos_texts=sample_positives,
        neg_texts=sample_negatives,
        model_name=args.model,
        pretrained=args.pretrained,
        target_token=args.target_token,
        output_dir=args.output_dir,
        device=device
    )
