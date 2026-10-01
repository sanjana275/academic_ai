"""
Model management and inference inspection for LLM X-Ray.
Handles loading Hugging Face models, inspecting forward passes,
extracting embeddings, attention weights, hidden states, and next-token distributions.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit as st


@st.cache_resource(show_spinner=False)
def load_model(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """
    Bulletproof multi-strategy loader.
    Tries 4 combinations (eager/default attn x full-RAM/device_map) so one always works.
    Uses float32 on CPU to avoid NaN issues with float16 on CPU-only machines.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    last_error = None

    # Strategy A: load fully into RAM then .to(device) -- most reliable
    for attn_impl in ("eager", None):
        try:
            kwargs = dict(torch_dtype=dtype, trust_remote_code=True)
            if attn_impl:
                kwargs["attn_implementation"] = attn_impl
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            model = model.to(device)
            model.eval()
            return model, device
        except Exception as e:
            last_error = e

    # Strategy B: device_map forces all layers to one device (no disk offload)
    for attn_impl in ("eager", None):
        try:
            kwargs = dict(
                torch_dtype=dtype,
                device_map={"": device},
                trust_remote_code=True,
            )
            if attn_impl:
                kwargs["attn_implementation"] = attn_impl
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            model.eval()
            return model, device
        except Exception as e:
            last_error = e

    raise RuntimeError(
        f"Could not load model '{model_name}'. Last error: {last_error}"
    )


def get_model_metadata(model) -> Dict[str, Any]:
    """
    Extracts architectural metadata from model config.
    """
    config = model.config
    num_params = sum(p.numel() for p in model.parameters())

    def format_num(n: int) -> str:
        if n >= 1e9:
            return f"{n / 1e9:.2f}B"
        elif n >= 1e6:
            return f"{n / 1e6:.2f}M"
        return f"{n:,}"

    num_layers = getattr(config, "num_hidden_layers", 28)
    hidden_size = getattr(config, "hidden_size", 1536)
    num_heads = getattr(config, "num_attention_heads", 12)
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    vocab_size = getattr(config, "vocab_size", 151936)
    intermediate_size = getattr(config, "intermediate_size", 8960)
    max_position = getattr(config, "max_position_embeddings", 32768)

    return {
        "num_params": num_params,
        "num_params_formatted": format_num(num_params),
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": hidden_size // num_heads if num_heads else 128,
        "vocab_size": vocab_size,
        "intermediate_size": intermediate_size,
        "max_position_embeddings": max_position,
        "model_type": getattr(config, "model_type", "qwen2"),
        "torch_dtype": str(model.dtype).replace("torch.", "")
    }


def extract_embeddings(model, input_ids_tensor: torch.Tensor, device: str) -> Dict[str, Any]:
    """
    Extracts the input token embeddings from the model's embedding layer.
    """
    input_ids = input_ids_tensor.to(device)
    embedding_layer = model.get_input_embeddings()

    with torch.no_grad():
        embeddings = embedding_layer(input_ids)  # (1, seq_len, hidden_dim)

    embed_np = embeddings[0].detach().cpu().to(torch.float32).numpy()
    
    # Calculate statistics across embedding vectors
    stats = {
        "dimension": embed_np.shape[-1],
        "shape": list(embed_np.shape),
        "min": float(np.min(embed_np)),
        "max": float(np.max(embed_np)),
        "mean": float(np.mean(embed_np)),
        "std": float(np.std(embed_np)),
        "norms": np.linalg.norm(embed_np, axis=-1).tolist()
    }

    return {
        "embeddings": embed_np,
        "stats": stats
    }


def run_xray_forward(
    model,
    tokenizer,
    input_ids_tensor: torch.Tensor,
    device: str,
    top_k_candidates: int = 10,
    temperature: float = 1.0
) -> Dict[str, Any]:
    """
    Executes a single forward pass with output_attentions=True and output_hidden_states=True.
    Returns:
    - attentions: list of 2D/3D numpy arrays for each layer
    - hidden_states: list of 2D numpy arrays for each layer (seq_len, hidden_dim)
    - next_token_logits: logits for last position
    - top_k_tokens: list of (token_str, prob, id)
    - entropy: Shannon entropy of the next-token probability distribution
    """
    input_ids = input_ids_tensor.to(device)
    
    # Ensure eager attention is enabled so attention matrices are returned
    if hasattr(model, "set_attn_implementation"):
        try:
            model.set_attn_implementation("eager")
        except Exception:
            pass

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True
        )

    # 1. Attentions: list of len(num_layers), each shape (num_heads, seq_len, seq_len)
    attentions = []
    if outputs.attentions is not None:
        for layer_attn in outputs.attentions:
            # layer_attn shape: (batch_size=1, num_heads, seq_len, seq_len)
            attn_cpu = layer_attn[0].detach().cpu().to(torch.float32).numpy()
            attentions.append(attn_cpu)

    # 2. Hidden states: list of len(num_layers + 1), each shape (seq_len, hidden_dim)
    hidden_states = []
    if outputs.hidden_states is not None:
        for hs in outputs.hidden_states:
            # hs shape: (batch_size=1, seq_len, hidden_dim)
            hs_cpu = hs[0].detach().cpu().to(torch.float32).numpy()
            hidden_states.append(hs_cpu)

    # 3. Logits and probabilities for predicting the next token
    # Last position logits
    last_logits = outputs.logits[0, -1, :].detach().cpu().to(torch.float32)

    # Apply temperature
    temp = max(1e-4, temperature)
    scaled_logits = last_logits / temp
    probs = F.softmax(scaled_logits, dim=-1)

    # Compute entropy: H(P) = - sum(p * log2(p))
    valid_probs = probs[probs > 1e-12]
    entropy = float(-(valid_probs * torch.log2(valid_probs)).sum().item())

    # Top-K candidate extraction
    top_values, top_indices = torch.topk(probs, k=min(top_k_candidates, len(probs)))
    special_ids_set = set(getattr(tokenizer, "all_special_ids", []))
    special_tokens_set = set(getattr(tokenizer, "all_special_tokens", []))
    
    top_candidates = []
    for val, idx in zip(top_values.tolist(), top_indices.tolist()):
        token_str = tokenizer.decode([idx])
        is_sp = (idx in special_ids_set) or (token_str in special_tokens_set)
        # Clean representation
        if token_str == "\n":
            display_str = "\\n"
        elif token_str.isspace():
            display_str = repr(token_str)
        else:
            display_str = token_str

        top_candidates.append({
            "token": display_str,
            "raw_token": token_str,
            "id": idx,
            "prob": float(val),
            "is_special": is_sp
        })

    return {
        "attentions": attentions,
        "hidden_states": hidden_states,
        "last_logits": last_logits.numpy(),
        "top_candidates": top_candidates,
        "entropy": entropy
    }


def generate_autoregressive_trace(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 35,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    status_callback=None,
    use_chat_template: bool = True,
    system_prompt: Optional[str] = "You are a helpful, knowledgeable AI assistant. Provide clear, direct, and accurate answers."
) -> Dict[str, Any]:
    """
    Step-by-step autoregressive generation that records every token generated,
    along with candidate distributions, probabilities, and special token indicators.
    Supports chat templates for instruct models so prompts receive direct, intelligent answers.
    """
    if use_chat_template and hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        formatted_prompt = prompt

    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    prompt_len = input_ids.shape[1]

    generated_ids = input_ids.clone()
    generation_steps = []
    
    stop_token_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_token_ids.add(tokenizer.eos_token_id)
    if getattr(tokenizer, "pad_token_id", None) is not None:
        stop_token_ids.add(tokenizer.pad_token_id)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and isinstance(im_end_id, int) and im_end_id > 0:
        stop_token_ids.add(im_end_id)
    special_ids_set = set(getattr(tokenizer, "all_special_ids", [])) | stop_token_ids

    stop_reason = f"Max new tokens ({max_new_tokens}) reached"
    current_text = ""

    for step_i in range(1, max_new_tokens + 1):
        if status_callback:
            status_callback(step_i, max_new_tokens)

        with torch.no_grad():
            outputs = model(input_ids=generated_ids, return_dict=True)
            last_logits = outputs.logits[0, -1, :].detach().cpu().to(torch.float32)

        # Apply temperature
        temp = max(1e-4, temperature)
        scaled_logits = last_logits / temp

        # Apply Top-K filtering
        if top_k > 0:
            topk_vals, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
            min_val = topk_vals[-1]
            scaled_logits = torch.where(scaled_logits < min_val, torch.full_like(scaled_logits, -float("Inf")), scaled_logits)

        # Apply Top-P (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift right to keep at least one token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            scaled_logits[indices_to_remove] = -float("Inf")

        probs = F.softmax(scaled_logits, dim=-1)

        # Candidate top-5 for visualization
        top_vals, top_idxs = torch.topk(probs, k=5)
        top_candidates = []
        for val, idx in zip(top_vals.tolist(), top_idxs.tolist()):
            dec = tokenizer.decode([idx])
            top_candidates.append({
                "token": dec if dec.strip() else repr(dec),
                "id": idx,
                "prob": float(val),
                "is_special": (idx in special_ids_set)
            })

        # Calculate step entropy
        valid_probs = probs[probs > 1e-12]
        step_entropy = float(-(valid_probs * torch.log2(valid_probs)).sum().item())

        # Sample or greedy
        if temperature < 0.05:
            next_token_id = int(torch.argmax(probs).item())
        else:
            next_token_id = int(torch.multinomial(probs, num_samples=1).item())

        chosen_prob = float(probs[next_token_id].item())
        chosen_str = tokenizer.decode([next_token_id])

        # If stop token, record termination step and break
        if next_token_id in stop_token_ids:
            generation_steps.append({
                "step": step_i,
                "token": chosen_str if chosen_str.strip() else repr(chosen_str),
                "raw_token": chosen_str,
                "id": next_token_id,
                "prob": chosen_prob,
                "top_candidates": top_candidates,
                "entropy": step_entropy,
                "cumulative_response": current_text,
                "is_special": True,
                "role": "EOS (End of Sequence - Generation Completed)"
            })
            stop_reason = f"Special EOS Token ({chosen_str.strip() or f'ID {next_token_id}'}) generated"
            break

        # Append next token
        next_tensor = torch.tensor([[next_token_id]], device=device)
        generated_ids = torch.cat([generated_ids, next_tensor], dim=-1)

        # Decode entire sequence so far
        current_text = tokenizer.decode(generated_ids[0][prompt_len:], skip_special_tokens=True).strip()

        generation_steps.append({
            "step": step_i,
            "token": chosen_str if chosen_str.strip() else repr(chosen_str),
            "raw_token": chosen_str,
            "id": next_token_id,
            "prob": chosen_prob,
            "top_candidates": top_candidates,
            "entropy": step_entropy,
            "cumulative_response": current_text,
            "is_special": (next_token_id in special_ids_set),
            "role": "Standard Token"
        })

    full_response = tokenizer.decode(generated_ids[0][prompt_len:], skip_special_tokens=True).strip()
    
    return {
        "full_response": full_response,
        "generation_steps": generation_steps,
        "total_new_tokens": len(generation_steps),
        "all_input_ids": generated_ids,
        "formatted_prompt": formatted_prompt,
        "stop_reason": stop_reason
    }
