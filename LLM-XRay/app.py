"""
LLM X-Ray: Interactive Visualization of LLM Inference
Main Streamlit Application.
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
from typing import Optional

from tokenizer import load_tokenizer, tokenize_text, format_token_chips_markdown
from model import (
    load_model,
    get_model_metadata,
    extract_embeddings,
    run_xray_forward,
    generate_autoregressive_trace
)
from visualization import (
    plot_embeddings_pca,
    plot_embedding_vector_slice,
    plot_attention_heatmap,
    plot_hidden_states_pca_trajectory,
    plot_layer_cosine_similarity,
    plot_top_k_probabilities,
    plot_autoregressive_timeline
)


# Configure Page
st.set_page_config(
    page_title="LLM X-Ray — Interactive LLM Inference",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    /* Global Styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Header Gradient & Badges */
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .hero-sub {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 20px;
    }
    .flow-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 2px 4px;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Stats Card */
    .stat-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        backdrop-filter: blur(8px);
    }
    .stat-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .stat-desc {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 4px;
    }
    
    /* Pipeline Step Box */
    .pipeline-step {
        border-left: 3px solid #6366f1;
        padding-left: 14px;
        margin: 12px 0;
    }
    
    /* Clean Tab Headers */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0px 0px;
        padding: 8px 16px;
        font-weight: 600;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔬 **LLM X-Ray Controls**")
    st.markdown("Inspect every step of inference inside a real Hugging Face model.")
    
    # Model Selection
    model_choice = st.selectbox(
        "Model Selection",
        [
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Custom..."
        ],
        index=0,
        help="Default is Qwen2.5-1.5B-Instruct. 0.5B is also available for ultra-fast response on CPU."
    )

    if model_choice == "Custom...":
        selected_model_name = st.text_input("Hugging Face Model ID", value="Qwen/Qwen2.5-1.5B-Instruct")
    else:
        selected_model_name = model_choice

    st.markdown("---")
    st.markdown("#### ⚙️ **Generation Parameters**")
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.05,
                            help="Higher values make output more random, 0.0 is deterministic (greedy).")
    max_new_tokens = st.slider("Max New Tokens", min_value=1, max_value=50, value=15, step=1,
                               help="Number of tokens to generate step-by-step.")
    top_k = st.slider("Top-K", min_value=0, max_value=100, value=50, step=5,
                      help="Keep only top K tokens with highest probability (0 to disable).")
    top_p = st.slider("Top-P (Nucleus)", min_value=0.1, max_value=1.0, value=0.9, step=0.05,
                      help="Keep cumulative probability up to P.")

    st.markdown("---")
    st.markdown("#### 🔍 **Inspection Controls**")
    top_k_candidates_inspect = st.slider("Top Predictions to Show", min_value=3, max_value=20, value=8)

    st.markdown("---")
    # Quick Status & Device
    device_name = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    st.info(f"**Runtime Device:** `{device_name}`\n\nNo external API required — 100% local inference.")


# -------------------------------------------------------------
# MAIN APP HEADER
# -------------------------------------------------------------
st.markdown('<div class="hero-title">LLM X-Ray: Inside the Transformer</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Interactive visual breakdown of autoregressive inference: Tokenization ➔ Embeddings ➔ Attention ➔ Hidden States ➔ Next-Token Logits ➔ Response</div>', unsafe_allow_html=True)

# Visual Pipeline Flow Badges
st.markdown("""
<div style="margin-bottom: 22px;">
    <span class="flow-badge">1. Input Text</span> ➔
    <span class="flow-badge">2. Tokenizer</span> ➔
    <span class="flow-badge">3. Token IDs</span> ➔
    <span class="flow-badge">4. Input Embeddings</span> ➔
    <span class="flow-badge">5. 28 Transformer Layers</span> ➔
    <span class="flow-badge">6. Multi-Head Attention</span> ➔
    <span class="flow-badge">7. LM Head & Logits</span> ➔
    <span class="flow-badge">8. Softmax Probabilities</span> ➔
    <span class="flow-badge">9. Next Token</span>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# MODEL & TOKENIZER LOADING
# -------------------------------------------------------------
with st.spinner(f"Loading '{selected_model_name}' weights into local memory..."):
    try:
        tokenizer = load_tokenizer(selected_model_name)
        model, device = load_model(selected_model_name)
        model_meta = get_model_metadata(model)
        model_loaded = True
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Tip: Check internet connectivity if downloading for the first time, or select `Qwen/Qwen2.5-0.5B-Instruct`.")
        model_loaded = False


if model_loaded:
    # Top Metrics Row
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    with col_m1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-title">Parameters</div>
            <div class="stat-value">{model_meta['num_params_formatted']}</div>
            <div class="stat-desc">Total model weights</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-title">Transformer Layers</div>
            <div class="stat-value">{model_meta['num_layers']}</div>
            <div class="stat-desc">Decoder blocks</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-title">Hidden Dim (d_model)</div>
            <div class="stat-value">{model_meta['hidden_size']}</div>
            <div class="stat-desc">Vector dimension</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-title">Attention Heads</div>
            <div class="stat-value">{model_meta['num_heads']}</div>
            <div class="stat-desc">{model_meta['num_kv_heads']} KV heads (GQA)</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m5:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-title">Vocab Size</div>
            <div class="stat-value">{model_meta['vocab_size']:,}</div>
            <div class="stat-desc">Byte-Pair tokens</div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # PROMPT INPUT SECTION
    # -------------------------------------------------------------
    st.markdown("### 💬 **Prompt Input**")
    
    # Preset quick buttons
    preset_cols = st.columns([1.5, 1.5, 1.8, 1.8, 4])
    default_prompt = "What is AI?"
    
    if "prompt_input" not in st.session_state:
        st.session_state["prompt_input"] = default_prompt

    with preset_cols[0]:
        if st.button("💡 What is AI?"):
            st.session_state["prompt_input"] = "What is AI?"
    with preset_cols[1]:
        if st.button("🇫🇷 Capital of France"):
            st.session_state["prompt_input"] = "The capital of France is"
    with preset_cols[2]:
        if st.button("🎭 To be or not to be"):
            st.session_state["prompt_input"] = "To be or not to be, that is the"
    with preset_cols[3]:
        if st.button("🐍 Fibonacci Python"):
            st.session_state["prompt_input"] = "def fibonacci(n):"

    prompt_text = st.text_area(
        "Enter your prompt to send through the LLM X-Ray:",
        value=st.session_state["prompt_input"],
        height=70,
        placeholder="Type a sentence or question here..."
    )

    col_btn, col_info = st.columns([2, 5])
    with col_btn:
        run_inference = st.button("⚡ Run LLM X-Ray & Generate", type="primary", use_container_width=True)

    # Store state across reruns so user can explore layers/heads without recomputing
    if "xray_data" not in st.session_state or run_inference:
        if run_inference or "xray_data" not in st.session_state:
            with st.spinner("Extracting token embeddings, running Transformer forward pass, and sampling tokens..."):
                # 1. Tokenize
                tok_res = tokenize_text(tokenizer, prompt_text)
                input_ids_tensor = tok_res["encoding"]["input_ids"]

                # 2. Extract Embeddings
                embed_res = extract_embeddings(model, input_ids_tensor, device)

                # 3. Forward pass for Prompt inspection (Attentions, Hidden States, Logits)
                fwd_res = run_xray_forward(
                    model,
                    tokenizer,
                    input_ids_tensor,
                    device,
                    top_k_candidates=top_k_candidates_inspect,
                    temperature=temperature
                )

                # 4. Autoregressive step-by-step generation
                gen_progress = st.progress(0, text="Generating autoregressive response...")
                def update_progress(current_step, total):
                    gen_progress.progress(current_step / total, text=f"Generated token {current_step}/{total}...")

                gen_res = generate_autoregressive_trace(
                    model,
                    tokenizer,
                    prompt_text,
                    device,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    status_callback=update_progress
                )
                gen_progress.empty()

                st.session_state["xray_data"] = {
                    "prompt": prompt_text,
                    "tok_res": tok_res,
                    "embed_res": embed_res,
                    "fwd_res": fwd_res,
                    "gen_res": gen_res
                }

    data = st.session_state["xray_data"]
    tok_res = data["tok_res"]
    embed_res = data["embed_res"]
    fwd_res = data["fwd_res"]
    gen_res = data["gen_res"]

    # Quick Response Banner
    st.markdown("---")
    st.markdown("#### 🤖 **Generated Output**")
    col_out_text, col_out_meta = st.columns([3, 1])
    with col_out_text:
        st.success(f"**Prompt:** *{data['prompt']}*\n\n**Response:** {gen_res['full_response']}")
    with col_out_meta:
        st.metric("New Tokens Generated", gen_res['total_new_tokens'])
        st.metric("Next Token Uncertainty (Entropy)", f"{fwd_res['entropy']:.2f} bits")

    # -------------------------------------------------------------
    # DETAILED X-RAY TABS
    # -------------------------------------------------------------
    tabs = st.tabs([
        "🔤 1. Tokenization",
        "🧬 2. Embeddings & PCA",
        "🏗️ 3. Transformer Layers",
        "🔍 4. Multi-Head Attention",
        "📊 5. Hidden States Depth",
        "🎯 6. Logits & Probabilities",
        "⏱️ 7. Token-by-Token Tracer"
    ])

    # -------------------------------------------------------------
    # TAB 1: TOKENIZATION
    # -------------------------------------------------------------
    with tabs[0]:
        st.markdown("### Step 4 — Tokenization Inspection")
        st.markdown("""
        Before an LLM processes text, words and subwords are broken down by a **Byte-Pair Encoding (BPE)** tokenizer 
        into discrete tokens, which map to unique integer **Token IDs** in the model's vocabulary table.
        """)

        st.markdown(f"**Total Tokens:** `{tok_res['num_tokens']}`")
        
        # Visual chips representation
        st.markdown("#### Visual Token Chips:")
        chips_html = format_token_chips_markdown(tok_res["display_tokens"], tok_res["input_ids"])
        st.markdown(chips_html, unsafe_allow_html=True)

        st.markdown("#### Detailed Token Breakdown:")
        df_tokens = pd.DataFrame(tok_res["token_info"])
        st.dataframe(df_tokens, use_container_width=True, hide_index=True)

        st.markdown("#### Token ID Flow Representation:")
        flow_tokens = " | ".join(tok_res["display_tokens"])
        flow_ids = " | ".join(str(i) for i in tok_res["input_ids"])
        st.code(f"Input:    {data['prompt']}\nTokens:   {flow_tokens}\nToken IDs:{flow_ids}", language="text")

    # -------------------------------------------------------------
    # TAB 2: EMBEDDINGS
    # -------------------------------------------------------------
    with tabs[1]:
        st.markdown("### Step 5 — Input Embeddings & 2D PCA Visualization")
        st.markdown("""
        Each Token ID looks up a dense vector in the embedding matrix (`nn.Embedding(vocab_size, hidden_dim)`).
        This transforms discrete token numbers into a continuous high-dimensional space where semantic relationships reside.
        """)

        embed_matrix = embed_res["embeddings"]  # (seq_len, hidden_dim)
        stats = embed_res["stats"]

        c_e1, c_e2, c_e3, c_e4 = st.columns(4)
        c_e1.metric("Embedding Dimension", stats["dimension"])
        c_e2.metric("Mean Value", f"{stats['mean']:.4f}")
        c_e3.metric("Std Deviation", f"{stats['std']:.4f}")
        c_e4.metric("Value Range", f"[{stats['min']:.2f}, {stats['max']:.2f}]")

        col_pca, col_vec = st.columns([1, 1])
        with col_pca:
            st.markdown("#### 2D PCA of Token Embeddings")
            st.caption("Principal Component Analysis projects the 1,536-dimensional vectors onto 2 principal axes.")
            pca_fig = plot_embeddings_pca(tok_res["display_tokens"], tok_res["input_ids"], embed_matrix)
            st.plotly_chart(pca_fig, use_container_width=True)

        with col_vec:
            st.markdown("#### Inspect Specific Token Vector")
            selected_tok_idx = st.selectbox(
                "Select Token to Inspect Vector Values:",
                range(len(tok_res["display_tokens"])),
                format_func=lambda i: f"[{i}] '{tok_res['display_tokens'][i]}' (ID: {tok_res['input_ids'][i]})"
            )
            vec = embed_matrix[selected_tok_idx]
            st.markdown(f"**L2 Norm (Magnitude):** `{np.linalg.norm(vec):.3f}`")
            vec_fig = plot_embedding_vector_slice(
                tok_res["display_tokens"][selected_tok_idx],
                tok_res["input_ids"][selected_tok_idx],
                vec,
                max_dims=128
            )
            st.plotly_chart(vec_fig, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 3: TRANSFORMER ARCHITECTURE
    # -------------------------------------------------------------
    with tabs[2]:
        st.markdown("### Step 6 — Transformer Layer Architecture")
        st.markdown("""
        The embedding vectors pass sequentially through **28 Transformer Decoder blocks**.
        Each block refines the representation of each token by attending to previous tokens and running a non-linear feedforward MLP.
        """)

        col_arch_diag, col_arch_info = st.columns([1, 1])

        with col_arch_diag:
            st.markdown("#### Transformer Architecture Flow:")
            st.markdown("""
            ```
            Input Tokens: ['What', ' is', ' AI', '?']
                     │
                     ▼
            [ Embedding Layer ] (151,936 vocab ➔ 1,536 hidden_dim)
                     │
                     ▼
            ┌──────────────────────────────────────────────┐
            │  Transformer Layer 0                         │
            │  ├── Input RMSNorm                           │
            │  ├── Grouped-Query Attention (12 Q, 2 KV)    │
            │  │   └── Rotary Position Embeddings (RoPE)   │
            │  ├── Residual Add (+)                        │
            │  ├── Post-Attention RMSNorm                  │
            │  ├── SwiGLU MLP (1,536 ➔ 8,960 ➔ 1,536)     │
            │  └── Residual Add (+)                        │
            └──────────────────────────────────────────────┘
                     │
                     ▼
            ┌──────────────────────────────────────────────┐
            │  Transformer Layer 1 ... 27 (Repeated 28x)   │
            └──────────────────────────────────────────────┘
                     │
                     ▼
            [ Final RMSNorm ]
                     │
                     ▼
            [ Language Modeling Head (Linear) ]
                     │
                     ▼
            Logits (151,936 vocabulary scores)
                     │
                     ▼
            [ Softmax ] ➔ Next-Token Probability Distribution
            ```
            """)

        with col_arch_info:
            st.markdown("#### Architectural Specifications:")
            specs_df = pd.DataFrame([
                {"Property": "Architecture Class", "Value": model_meta["model_type"].capitalize() + "ForCausalLM"},
                {"Property": "Total Parameters", "Value": model_meta["num_params_formatted"]},
                {"Property": "Transformer Decoder Layers", "Value": model_meta["num_layers"]},
                {"Property": "Hidden Dimension (d_model)", "Value": model_meta["hidden_size"]},
                {"Property": "Query Attention Heads", "Value": model_meta["num_heads"]},
                {"Property": "Key / Value Attention Heads", "Value": model_meta["num_kv_heads"]},
                {"Property": "Head Dimension", "Value": model_meta["head_dim"]},
                {"Property": "Intermediate MLP Size", "Value": model_meta["intermediate_size"]},
                {"Property": "Activation Function", "Value": "SiLU (SwiGLU)"},
                {"Property": "Positional Encoding", "Value": "RoPE (Rotary Position Embeddings)"},
                {"Property": "Normalization", "Value": "RMSNorm"},
                {"Property": "Context Window (Max Tokens)", "Value": f"{model_meta['max_position_embeddings']:,}"},
                {"Property": "Inference Precision", "Value": model_meta["torch_dtype"]}
            ])
            st.dataframe(specs_df, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------
    # TAB 4: ATTENTION HEATMAP
    # -------------------------------------------------------------
    with tabs[3]:
        st.markdown("### Step 7 — Multi-Head Attention Visualization")
        st.markdown("""
        In self-attention, each query token assigns an attention weight $\\in [0, 1]$ to all preceding tokens (causally masked).
        This determines how much context a token borrows from other tokens in the prompt.
        """)

        attentions = fwd_res.get("attentions", [])
        num_layers = len(attentions)

        if num_layers == 0:
            st.warning("⚠️ Attention weights could not be captured for this model. Eager attention mode is required to visualize attention heatmaps.")
        else:
            num_heads = attentions[0].shape[0] if num_layers > 0 else 12
            default_layer = min(12, max(0, num_layers - 1))

            col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1.5, 1.5, 1])
            with col_ctrl1:
                sel_layer = st.slider("Select Transformer Layer", 0, num_layers - 1, value=default_layer,
                                      help="Early layers capture syntax and punctuation; deeper layers capture semantics and task logic.")
            with col_ctrl2:
                head_options = ["Average Across All Heads"] + [f"Head {h}" for h in range(num_heads)]
                sel_head_opt = st.selectbox("Select Attention Head", head_options, index=0)
            with col_ctrl3:
                colorscale = st.selectbox("Heatmap Color", ["Viridis", "Plasma", "Blues", "Inferno", "Rocket"], index=0)

            # Safely clamp layer index
            safe_layer = min(max(0, sel_layer), num_layers - 1)
            layer_attn_all_heads = attentions[safe_layer]  # (num_heads, seq_len, seq_len)
            
            if sel_head_opt == "Average Across All Heads":
                attn_matrix = np.mean(layer_attn_all_heads, axis=0)
                head_idx_val = None
            else:
                head_idx_val = int(sel_head_opt.replace("Head ", ""))
                attn_matrix = layer_attn_all_heads[head_idx_val]

            attn_fig = plot_attention_heatmap(
                attention_matrix=attn_matrix,
                tokens=tok_res["display_tokens"],
                layer_idx=safe_layer,
                head_idx=head_idx_val,
                colorscale=colorscale
            )
            st.plotly_chart(attn_fig, use_container_width=True)

            st.caption("ℹ️ **Interpretation:** In causal decoder models, the upper triangle is 0 (tokens cannot attend to future tokens). Higher weights indicate stronger semantic connection.")

    # -------------------------------------------------------------
    # TAB 5: HIDDEN STATES DEPTH
    # -------------------------------------------------------------
    with tabs[4]:
        st.markdown("### Step 8 — Hidden States & Representation Drift Across Layers")
        st.markdown("""
        As tokens pass from Layer 0 (Embedding) through deeper layers, their internal vector representation 
        gradually shifts from surface-level lexical properties to abstract contextual semantics.
        """)

        hidden_states = fwd_res.get("hidden_states", [])

        if not hidden_states:
            st.warning("⚠️ Hidden states were not captured for this model.")
        else:
            col_h1, col_h2 = st.columns([1, 1])
            with col_h1:
                st.markdown("#### Hidden State Trajectory (PCA)")
                st.caption(f"Tracks how the selected token's state vector drifts across all {len(hidden_states)} layers.")
                sel_traj_tok_idx = st.selectbox(
                    "Select Token to Track Through Depth:",
                    range(len(tok_res["display_tokens"])),
                    index=len(tok_res["display_tokens"]) - 1,
                    format_func=lambda i: f"[{i}] '{tok_res['display_tokens'][i]}'",
                    key="traj_tok_select"
                )
                traj_fig = plot_hidden_states_pca_trajectory(
                    hidden_states_by_layer=hidden_states,
                    tokens=tok_res["display_tokens"],
                    selected_token_idx=sel_traj_tok_idx
                )
                st.plotly_chart(traj_fig, use_container_width=True)

            with col_h2:
                st.markdown("#### Layer-to-Layer Cosine Similarity")
                st.caption("Measures how rapidly token representations transform between consecutive layers.")
                sim_fig = plot_layer_cosine_similarity(
                    hidden_states_by_layer=hidden_states,
                    selected_token_idx=sel_traj_tok_idx
                )
                st.plotly_chart(sim_fig, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 6: LOGITS & PROBABILITIES
    # -------------------------------------------------------------
    with tabs[5]:
        st.markdown("### Step 9 — LM Head, Logits & Next-Token Probabilities")
        st.markdown("""
        The final hidden state of the prompt's last token is projected by the **LM Head** onto the entire vocabulary (151,936 items).
        Applying **Softmax** produces the probability distribution from which the next token is sampled.
        """)

        top_candidates = fwd_res["top_candidates"]
        top_tokens = [c["token"] for c in top_candidates]
        top_probs = [c["prob"] for c in top_candidates]
        top_ids = [c["id"] for c in top_candidates]

        first_gen_token = gen_res["generation_steps"][0]["token"] if gen_res["generation_steps"] else None

        prob_fig = plot_top_k_probabilities(
            top_tokens=top_tokens,
            top_probs=top_probs,
            top_ids=top_ids,
            selected_token=first_gen_token,
            title="Top Candidates for Next Token Prediction"
        )
        st.plotly_chart(prob_fig, use_container_width=True)

        # Candidate table
        st.markdown("#### Top Candidate Rankings:")
        cand_df = pd.DataFrame([
            {
                "Rank": i + 1,
                "Candidate Token": c["token"],
                "Token ID": c["id"],
                "Probability (%)": f"{c['prob'] * 100:.2f}%",
                "Logit Score": f"{fwd_res['last_logits'][c['id']]:.2f}",
                "Selected": "✅ Chosen" if first_gen_token and c["token"] == first_gen_token else ""
            }
            for i, c in enumerate(top_candidates)
        ])
        st.dataframe(cand_df, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------
    # TAB 7: TOKEN-BY-TOKEN TRACER
    # -------------------------------------------------------------
    with tabs[6]:
        st.markdown("### Step 10 — Step-by-Step Autoregressive Generation Tracer")
        st.markdown("""
        LLMs generate responses **autoregressively**: one token at a time. Each newly generated token is appended 
        back into the input sequence to predict the subsequent token, repeating until an `<|im_end|>` or `<|endoftext|>` token is emitted.
        """)

        steps = gen_res["generation_steps"]

        if not steps:
            st.warning("No new tokens generated. Check your prompt or max new tokens setting.")
        else:
            # Timeline Confidence Chart
            timeline_fig = plot_autoregressive_timeline(steps)
            st.plotly_chart(timeline_fig, use_container_width=True)

            # Interactive Step-by-Step Inspector
            st.markdown("#### 🔍 Replay Generation Step-by-Step:")
            sel_step_num = st.slider("Select Step to Inspect Internals", min_value=1, max_value=len(steps), value=1)
            active_step_data = steps[sel_step_num - 1]

            col_s1, col_s2 = st.columns([1, 1])

            with col_s1:
                st.markdown(f"**Step {active_step_data['step']} Token Chosen:** `{active_step_data['token']}` (ID: `{active_step_data['id']}`)")
                st.markdown(f"**Model Confidence:** `{active_step_data['prob']*100:.2f}%`")
                st.markdown(f"**Step Uncertainty (Entropy):** `{active_step_data['entropy']:.2f} bits`")

                st.markdown("##### Cumulative Response at this Step:")
                highlighted_text = f"*{data['prompt']}* **{active_step_data['cumulative_response']}**"
                st.info(highlighted_text)

            with col_s2:
                st.markdown("##### Top Competing Alternatives at this Step:")
                step_cands = active_step_data["top_candidates"]
                step_fig = plot_top_k_probabilities(
                    top_tokens=[c["token"] for c in step_cands],
                    top_probs=[c["prob"] for c in step_cands],
                    top_ids=[c["id"] for c in step_cands],
                    selected_token=active_step_data["token"],
                    title=f"Distribution at Step {active_step_data['step']}"
                )
                st.plotly_chart(step_fig, use_container_width=True)

            # Cumulative Full Generation Sequence Table
            st.markdown("#### Generation History Table:")
            history_rows = []
            for s in steps:
                history_rows.append({
                    "Step": s["step"],
                    "Emitted Token": s["token"],
                    "Token ID": s["id"],
                    "Confidence": f"{s['prob']*100:.1f}%",
                    "Cumulative Output": s["cumulative_response"]
                })
            st.dataframe(pd.DataFrame(history_rows), use_container_width=True, hide_index=True)
