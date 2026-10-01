"""
Visualization module for LLM X-Ray.
Generates Plotly interactive charts for:
- 2D PCA of Token Embeddings
- Embedding Vector Heatmap / Inspection
- Transformer Layer Flow
- Attention Heatmap (Layer & Head selectable)
- Hidden States Layer Drift / PCA Trajectory
- Next-Token Top-K Logits & Probability Distribution
- Step-by-Step Autoregressive Generation Tracer
"""

from typing import List, Dict, Any, Optional
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA


DARK_TEMPLATE = "plotly_dark"
LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(15, 23, 42, 0.7)",
    plot_bgcolor="rgba(30, 41, 59, 0.5)",
    font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", color="#e2e8f0"),
    margin=dict(l=40, r=40, t=50, b=40),
)


def plot_embeddings_pca(
    tokens: List[str],
    token_ids: List[int],
    embeddings: np.ndarray,
    title: str = "2D PCA Projection of Input Token Embeddings"
) -> go.Figure:
    """
    Computes 2D PCA on token embedding vectors and plots an interactive scatter plot.
    """
    n_samples, n_features = embeddings.shape

    if n_samples < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Need at least 2 tokens to compute PCA projection.<br>Please enter a longer prompt.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color="#94a3b8")
        )
        fig.update_layout(**LAYOUT_DEFAULTS, title=title)
        return fig

    # If n_samples == 2, 2 components can be fit
    n_components = min(2, n_samples)
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(embeddings)

    if n_components == 1:
        x_coords = coords[:, 0]
        y_coords = np.zeros(n_samples)
        var_exp = f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)"
        y_label = "Fixed (1D PCA)"
    else:
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]
        var_exp = (
            f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%), "
            f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"
        )
        y_label = f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"

    norms = np.linalg.norm(embeddings, axis=1)

    hover_texts = [
        f"<b>Pos:</b> {i}<br><b>Token:</b> '{tok}'<br><b>ID:</b> {tid}<br><b>L2 Norm:</b> {norm:.3f}"
        for i, (tok, tid, norm) in enumerate(zip(tokens, token_ids, norms))
    ]

    fig = go.Figure()

    # Draw faint line showing token sequence flow
    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_coords,
        mode="lines",
        line=dict(color="rgba(148, 163, 184, 0.35)", width=2, dash="dot"),
        hoverinfo="none",
        showlegend=False
    ))

    # Scatter points for each token
    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_coords,
        mode="markers+text",
        text=[f"[{i}] {t}" for i, t in enumerate(tokens)],
        textposition="top center",
        textfont=dict(size=12, color="#f8fafc"),
        hoverinfo="text",
        hovertext=hover_texts,
        marker=dict(
            size=16,
            color=list(range(n_samples)),
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Token Position", len=0.7),
            line=dict(width=2, color="#ffffff")
        ),
        showlegend=False
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text=f"{title}<br><sup>Explained Variance: {var_exp}</sup>", font=dict(size=16)),
        xaxis=dict(title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(title=y_label, gridcolor="rgba(255,255,255,0.08)"),
        hoverlabel=dict(bgcolor="#1e293b", font_size=13, font_family="monospace")
    )
    return fig


def plot_embedding_vector_slice(
    token_str: str,
    token_id: int,
    vector: np.ndarray,
    max_dims: int = 128
) -> go.Figure:
    """
    Plots the values of an embedding vector (or first N dimensions) as a bar/area chart.
    """
    display_dim = min(len(vector), max_dims)
    slice_vec = vector[:display_dim]
    x_axis = list(range(display_dim))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_axis,
        y=slice_vec,
        marker=dict(
            color=slice_vec,
            colorscale="RdBu",
            cmid=0,
            showscale=False
        ),
        hovertemplate="Dim %{x}: %{y:.4f}<extra></extra>"
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=f"Embedding Vector Values for '{token_str}' (ID: {token_id}) [First {display_dim} of {len(vector)} dims]",
        xaxis=dict(title="Dimension Index", gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(title="Value", gridcolor="rgba(255,255,255,0.08)"),
        height=320
    )
    return fig


def plot_attention_heatmap(
    attention_matrix: Optional[np.ndarray] = None,
    tokens: Optional[List[str]] = None,
    layer_idx: int = 0,
    head_idx: Optional[int] = None,
    colorscale: str = "Viridis",
    attn_matrix: Optional[np.ndarray] = None
) -> go.Figure:
    """
    Plots an interactive attention heatmap.
    attention_matrix / attn_matrix: 2D array (seq_len, seq_len)
    tokens: list of token labels
    """
    matrix_input = attention_matrix if attention_matrix is not None else attn_matrix
    if matrix_input is None:
        return go.Figure()

    if tokens is None:
        tokens = [str(i) for i in range(matrix_input.shape[0])]

    seq_len = len(tokens)
    # Ensure correct shape
    matrix = matrix_input[:seq_len, :seq_len]

    head_label = f"Head {head_idx}" if head_idx is not None else "Average Across Heads"
    title = f"Self-Attention Matrix — Layer {layer_idx} ({head_label})"

    customdata = np.empty((seq_len, seq_len), dtype=object)
    for q_idx in range(seq_len):
        for k_idx in range(seq_len):
            customdata[q_idx, k_idx] = (
                f"<b>Query (Row {q_idx}):</b> '{tokens[q_idx]}'<br>"
                f"<b>Key (Col {k_idx}):</b> '{tokens[k_idx]}'<br>"
                f"<b>Attention Weight:</b> {matrix[q_idx, k_idx]:.4f} ({matrix[q_idx, k_idx]*100:.1f}%)"
            )

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=tokens,
        y=tokens,
        customdata=customdata,
        hovertemplate="%{customdata}<extra></extra>",
        colorscale=colorscale,
        zmin=0.0,
        zmax=max(0.1, float(np.max(matrix))),
        colorbar=dict(title="Weight", len=0.8)
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(
            title="Key Tokens (Attended to)",
            tickangle=-45,
            gridcolor="rgba(255,255,255,0.05)",
            side="bottom"
        ),
        yaxis=dict(
            title="Query Tokens (Attending from)",
            autorange="reversed",
            gridcolor="rgba(255,255,255,0.05)"
        ),
        width=700,
        height=620
    )
    return fig


def plot_hidden_states_pca_trajectory(
    hidden_states_by_layer: List[np.ndarray],
    tokens: List[str],
    selected_token_idx: int = -1
) -> go.Figure:
    """
    Computes PCA on the selected token's hidden state vector across all transformer layers (0..28)
    to reveal the 'representation trajectory' through model depth.
    hidden_states_by_layer: list of (seq_len, hidden_dim) arrays, length num_layers + 1.
    """
    num_layers = len(hidden_states_by_layer)
    seq_len = len(tokens)
    target_idx = (selected_token_idx + seq_len) % seq_len
    target_token = tokens[target_idx]

    # Collect vector for target_token across all layers
    layer_vectors = np.array([layer_hs[target_idx] for layer_hs in hidden_states_by_layer])
    
    pca = PCA(n_components=2, random_state=42)
    coords_2d = pca.fit_transform(layer_vectors)

    layer_names = ["Embedding (Layer 0)"] + [f"Layer {l}" for l in range(1, num_layers)]

    fig = go.Figure()

    # Trajectory line
    fig.add_trace(go.Scatter(
        x=coords_2d[:, 0],
        y=coords_2d[:, 1],
        mode="lines",
        line=dict(color="#38bdf8", width=3),
        name="Depth Flow",
        hoverinfo="none"
    ))

    # Layer markers
    hover_texts = [
        f"<b>{name}</b><br>Token: '{target_token}' (Pos {target_idx})<br>Norm: {np.linalg.norm(vec):.2f}"
        for name, vec in zip(layer_names, layer_vectors)
    ]

    fig.add_trace(go.Scatter(
        x=coords_2d[:, 0],
        y=coords_2d[:, 1],
        mode="markers+text",
        text=[f"L{i}" for i in range(num_layers)],
        textposition="top right",
        textfont=dict(size=10, color="#cbd5e1"),
        hoverinfo="text",
        hovertext=hover_texts,
        marker=dict(
            size=12,
            color=list(range(num_layers)),
            colorscale="Plasma",
            showscale=True,
            colorbar=dict(title="Layer Depth", len=0.8),
            line=dict(width=1.5, color="#ffffff")
        ),
        name="Layer State"
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(
            text=f"Hidden State Trajectory for Token: '{target_token}' across {num_layers-1} Transformer Layers",
            font=dict(size=16)
        ),
        xaxis=dict(
            title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
            gridcolor="rgba(255,255,255,0.08)"
        ),
        yaxis=dict(
            title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
            gridcolor="rgba(255,255,255,0.08)"
        ),
        showlegend=False,
        height=500
    )
    return fig


def plot_layer_cosine_similarity(
    hidden_states_by_layer: List[np.ndarray],
    selected_token_idx: int = -1
) -> go.Figure:
    """
    Computes cosine similarity matrix of a token's hidden state between all pairs of layers.
    """
    num_layers = len(hidden_states_by_layer)
    vectors = np.array([layer_hs[selected_token_idx] for layer_hs in hidden_states_by_layer])
    
    # Normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
    norm_vectors = vectors / norms
    sim_matrix = np.dot(norm_vectors, norm_vectors.T)

    labels = ["Emb"] + [f"L{i}" for i in range(1, num_layers)]

    fig = go.Figure(data=go.Heatmap(
        z=sim_matrix,
        x=labels,
        y=labels,
        colorscale="Blues",
        zmin=min(0.0, float(np.min(sim_matrix))),
        zmax=1.0,
        colorbar=dict(title="Cosine Sim", len=0.8),
        hovertemplate="Layer %{y} vs Layer %{x}: %{z:.3f}<extra></extra>"
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="Layer-to-Layer Representation Cosine Similarity",
        xaxis=dict(title="Layer", side="bottom"),
        yaxis=dict(title="Layer", autorange="reversed"),
        height=480
    )
    return fig


def plot_top_k_probabilities(
    top_tokens: List[str],
    top_probs: List[float],
    top_ids: List[int],
    selected_token: Optional[str] = None,
    is_special_list: Optional[List[bool]] = None,
    title: str = "Next-Token Probability Distribution (Top Candidates)"
) -> go.Figure:
    """
    Horizontal bar chart showing top-k token predictions and their softmax probabilities.
    Special tokens are highlighted in amber.
    """
    # Reverse so top candidate is at the top of the horizontal bar chart
    tokens_rev = top_tokens[::-1]
    probs_rev = [p * 100 for p in top_probs[::-1]]  # convert to percentage
    ids_rev = top_ids[::-1]
    special_rev = is_special_list[::-1] if is_special_list else [False] * len(tokens_rev)

    colors = []
    labels = []
    for t, is_sp in zip(tokens_rev, special_rev):
        if is_sp:
            colors.append("#f59e0b")  # amber for special tokens
            labels.append(f"⚡ {t}")
        elif selected_token and t == selected_token:
            colors.append("#22c55e")  # green highlight for sampled token
            labels.append(f"'{t}'")
        else:
            colors.append("#6366f1")  # indigo for candidates
            labels.append(f"'{t}'")

    hover_texts = [
        f"<b>Token:</b> '{t}' {'(SPECIAL)' if sp else ''}<br><b>ID:</b> {tid}<br><b>Probability:</b> {p:.2f}%"
        for t, tid, p, sp in zip(tokens_rev, ids_rev, probs_rev, special_rev)
    ]

    fig = go.Figure(go.Bar(
        x=probs_rev,
        y=labels,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(color="rgba(255,255,255,0.2)", width=1)
        ),
        text=[f"{p:.2f}%" for p in probs_rev],
        textposition="outside",
        hoverinfo="text",
        hovertext=hover_texts
    ))

    max_prob = max(probs_rev) if probs_rev else 100

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(
            title="Softmax Probability (%)",
            range=[0, min(100, max_prob * 1.25 + 5)],
            gridcolor="rgba(255,255,255,0.08)"
        ),
        yaxis=dict(
            title="Candidate Token",
            tickfont=dict(family="monospace", size=13)
        ),
        height=max(360, len(top_tokens) * 32)
    )
    return fig


def plot_autoregressive_timeline(generation_steps: List[Dict[str, Any]]) -> go.Figure:
    """
    Plots the probability of the selected token at each generation step.
    Shows model confidence along the generated sequence.
    Emphasizes terminal / special tokens (like <|im_end|>).
    """
    if not generation_steps:
        return go.Figure()

    step_nums = [s["step"] for s in generation_steps]
    tokens = [s["token"] for s in generation_steps]
    probs = [s["prob"] * 100 for s in generation_steps]
    entropies = [s.get("entropy", 0.0) for s in generation_steps]
    specials = [s.get("is_special", False) for s in generation_steps]

    marker_colors = ["#f59e0b" if sp else "#34d399" for sp in specials]
    marker_sizes = [14 if sp else 10 for sp in specials]

    hover_texts = [
        f"<b>Step {s}:</b> '{t}' {'[SPECIAL EOS]' if sp else ''}<br><b>Confidence:</b> {p:.1f}%<br><b>Entropy:</b> {e:.2f}"
        for s, t, p, e, sp in zip(step_nums, tokens, probs, entropies, specials)
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=step_nums,
        y=probs,
        mode="lines+markers+text",
        text=tokens,
        textposition="top center",
        textfont=dict(family="monospace", size=11),
        line=dict(color="#10b981", width=3),
        marker=dict(size=marker_sizes, color=marker_colors, line=dict(width=2, color="#ffffff")),
        hoverinfo="text",
        hovertext=hover_texts,
        name="Confidence (%)"
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="Autoregressive Generation Confidence Per Token (⚡ Special tokens in Amber)",
        xaxis=dict(title="Generation Step", gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(title="Confidence (%)", range=[0, 105], gridcolor="rgba(255,255,255,0.08)"),
        height=350,
        showlegend=False
    )
    return fig
