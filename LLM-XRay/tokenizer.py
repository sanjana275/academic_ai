"""
Tokenization module for LLM X-Ray.
Handles loading tokenizer, encoding text, extracting token IDs,
and decoding tokens cleanly with metadata.
"""

from typing import List, Dict, Any
from transformers import AutoTokenizer
import streamlit as st


@st.cache_resource(show_spinner=False)
def load_tokenizer(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    """
    Loads and caches the Hugging Face tokenizer.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer '{model_name}': {str(e)}")


def tokenize_text(tokenizer, text: str) -> Dict[str, Any]:
    """
    Tokenizes input text and returns rich information for visualization:
    - input_ids: tensor/list of integer IDs
    - tokens: list of token strings (preserving whitespace visualization)
    - display_tokens: human-readable token labels
    - token_info: list of dicts with index, token, id, and decoded text
    """
    if not text:
        return {
            "input_ids": [],
            "tokens": [],
            "display_tokens": [],
            "token_info": [],
            "num_tokens": 0
        }

    encoding = tokenizer(text, return_tensors="pt")
    input_ids = encoding["input_ids"][0].tolist()
    
    # Extract raw tokens
    raw_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    token_info = []
    display_tokens = []
    
    for idx, (tok_id, raw_tok) in enumerate(zip(input_ids, raw_tokens)):
        decoded_single = tokenizer.decode([tok_id])
        
        # Clean representation for display
        # Qwen/Byte-pair tokenizers use special characters like Ġ or  or raw unicode for spaces
        clean_tok = raw_tok
        if clean_tok.startswith("Ġ"):
            display_str = " " + clean_tok[1:]
        elif clean_tok.startswith(" "):
            display_str = " " + clean_tok[1:]
        elif clean_tok == "\n":
            display_str = "\\n"
        elif clean_tok == "\t":
            display_str = "\\t"
        elif decoded_single.isspace():
            display_str = repr(decoded_single)
        else:
            display_str = decoded_single
            
        display_tokens.append(display_str)
        token_info.append({
            "Index": idx,
            "Token": display_str,
            "Raw Token": raw_tok,
            "Token ID": tok_id,
            "Decoded String": decoded_single,
            "Char Length": len(decoded_single)
        })

    return {
        "input_ids": input_ids,
        "raw_tokens": raw_tokens,
        "display_tokens": display_tokens,
        "token_info": token_info,
        "num_tokens": len(input_ids),
        "encoding": encoding
    }


def format_token_chips_markdown(display_tokens: List[str], input_ids: List[int]) -> str:
    """
    Generates markdown / HTML representation of tokens as stylized chips.
    """
    chips = []
    palette = [
        ("rgba(99, 102, 241, 0.15)", "#818cf8", "#c7d2fe"),
        ("rgba(168, 85, 247, 0.15)", "#c084fc", "#e9d5ff"),
        ("rgba(236, 72, 153, 0.15)", "#f472b6", "#fbcfe8"),
        ("rgba(14, 165, 233, 0.15)", "#38bdf8", "#bae6fd"),
        ("rgba(34, 197, 94, 0.15)", "#4ade80", "#bbf7d0"),
        ("rgba(234, 179, 8, 0.15)", "#facc15", "#fef08a"),
    ]

    for i, (tok, tid) in enumerate(zip(display_tokens, input_ids)):
        bg, border, text_col = palette[i % len(palette)]
        escaped_tok = (tok.replace("<", "&lt;").replace(">", "&gt;"))
        chip_html = (
            f'<div style="display:inline-flex; flex-direction:column; align-items:center; '
            f'background:{bg}; border:1px solid {border}; border-radius:8px; '
            f'padding:4px 8px; margin:4px; font-family:monospace; min-width:48px;">'
            f'<span style="font-weight:600; color:{text_col}; font-size:14px;">{escaped_tok}</span>'
            f'<span style="font-size:10px; opacity:0.75; color:#94a3b8;">ID: {tid}</span>'
            f'</div>'
        )
        chips.append(chip_html)

    return f'<div style="display:flex; flex-wrap:wrap; gap:4px; margin:10px 0;">{"".join(chips)}</div>'
