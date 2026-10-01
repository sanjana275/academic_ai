"""
Tokenization module for LLM X-Ray.
Handles loading tokenizer, encoding text, extracting token IDs,
identifying special tokens, and decoding tokens cleanly with metadata.
"""

from typing import List, Dict, Any, Optional
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


def get_special_token_role(tokenizer, tok_id: int, tok_str: str) -> str:
    """
    Determines the semantic role of a special token.
    """
    if tok_id == getattr(tokenizer, "eos_token_id", None):
        return "EOS (End of Sequence)"
    if tok_id == getattr(tokenizer, "bos_token_id", None):
        return "BOS (Beginning of Sequence)"
    if tok_id == getattr(tokenizer, "pad_token_id", None):
        return "PAD (Padding)"
    if tok_id == getattr(tokenizer, "unk_token_id", None):
        return "UNK (Unknown)"
    
    clean_str = tok_str.strip()
    if clean_str == "<|im_start|>":
        return "Chat Turn Header (<|im_start|>)"
    elif clean_str == "<|im_end|>":
        return "Chat Turn Delimiter (<|im_end|>)"
    elif clean_str == "<|endoftext|>":
        return "Document Boundary (<|endoftext|>)"
    elif "vision" in clean_str:
        return "Vision / Multimodal Delimiter"
    elif "box" in clean_str or "quad" in clean_str or "ref" in clean_str:
        return "Grounding / Spatial Anchor"
    return "Control / Special Token"


def get_special_tokens_table(tokenizer) -> List[Dict[str, Any]]:
    """
    Returns an informative list of all special tokens registered with the tokenizer.
    """
    special_tokens = []
    seen_ids = set()

    all_special = getattr(tokenizer, "all_special_tokens", [])
    all_ids = getattr(tokenizer, "all_special_ids", [])

    for tok_str, tok_id in zip(all_special, all_ids):
        if tok_id not in seen_ids:
            seen_ids.add(tok_id)
            special_tokens.append({
                "Token": tok_str,
                "Token ID": tok_id,
                "Role / Function": get_special_token_role(tokenizer, tok_id, tok_str),
                "Category": "Special Delimiter"
            })

    # Sort by Token ID
    special_tokens.sort(key=lambda x: x["Token ID"])
    return special_tokens


def tokenize_text(tokenizer, text: str) -> Dict[str, Any]:
    """
    Tokenizes input text and returns rich information for visualization:
    - input_ids: tensor/list of integer IDs
    - tokens: list of token strings (preserving whitespace visualization)
    - display_tokens: human-readable token labels
    - token_info: list of dicts with index, token, id, is_special, and decoded text
    - num_special_tokens: count of special tokens present in input
    """
    if not text:
        return {
            "input_ids": [],
            "raw_tokens": [],
            "display_tokens": [],
            "token_info": [],
            "is_special_list": [],
            "num_tokens": 0,
            "num_special_tokens": 0
        }

    encoding = tokenizer(text, return_tensors="pt")
    input_ids = encoding["input_ids"][0].tolist()
    
    # Extract raw tokens
    raw_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    special_ids_set = set(getattr(tokenizer, "all_special_ids", []))
    special_tokens_set = set(getattr(tokenizer, "all_special_tokens", []))
    
    token_info = []
    display_tokens = []
    is_special_list = []
    
    for idx, (tok_id, raw_tok) in enumerate(zip(input_ids, raw_tokens)):
        decoded_single = tokenizer.decode([tok_id])
        is_special = (tok_id in special_ids_set) or (raw_tok in special_tokens_set) or (decoded_single in special_tokens_set)
        is_special_list.append(is_special)
        
        # Clean representation for display
        clean_tok = raw_tok
        if is_special:
            display_str = raw_tok if raw_tok in special_tokens_set else decoded_single
            role_desc = get_special_token_role(tokenizer, tok_id, display_str)
        elif clean_tok.startswith("Ġ"):
            display_str = " " + clean_tok[1:]
            role_desc = "Standard Word/Subword"
        elif clean_tok.startswith(" "):
            display_str = " " + clean_tok[1:]
            role_desc = "Standard Word/Subword"
        elif clean_tok == "\n":
            display_str = "\\n"
            role_desc = "Newline Character"
        elif clean_tok == "\t":
            display_str = "\\t"
            role_desc = "Tab Character"
        elif decoded_single.isspace():
            display_str = repr(decoded_single)
            role_desc = "Whitespace"
        else:
            display_str = decoded_single
            role_desc = "Standard Word/Subword"
            
        display_tokens.append(display_str)
        token_info.append({
            "Index": idx,
            "Token": display_str,
            "Raw Token": raw_tok,
            "Token ID": tok_id,
            "Is Special": "⚡ SPECIAL" if is_special else "Standard",
            "Role": role_desc,
            "Decoded String": decoded_single,
            "Char Length": len(decoded_single)
        })

    num_special_tokens = sum(1 for s in is_special_list if s)

    return {
        "input_ids": input_ids,
        "raw_tokens": raw_tokens,
        "display_tokens": display_tokens,
        "token_info": token_info,
        "is_special_list": is_special_list,
        "num_tokens": len(input_ids),
        "num_special_tokens": num_special_tokens,
        "encoding": encoding
    }


def format_token_chips_markdown(
    display_tokens: List[str],
    input_ids: List[int],
    is_special_list: Optional[List[bool]] = None
) -> str:
    """
    Generates markdown / HTML representation of tokens as stylized chips.
    Special tokens are emphasized with an amber glowing border and SPECIAL badge.
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
        is_sp = is_special_list[i] if is_special_list and i < len(is_special_list) else False
        escaped_tok = (tok.replace("<", "&lt;").replace(">", "&gt;"))
        
        if is_sp:
            chip_html = (
                f'<div style="display:inline-flex; flex-direction:column; align-items:center; '
                f'background:rgba(245, 158, 11, 0.18); border:2px solid #f59e0b; border-radius:8px; '
                f'padding:4px 9px; margin:4px; font-family:monospace; min-width:54px; box-shadow:0 0 10px rgba(245,158,11,0.25);">'
                f'<span style="font-weight:700; color:#fbbf24; font-size:14px;">{escaped_tok}</span>'
                f'<span style="font-size:10px; opacity:0.85; color:#fde68a;">ID: {tid}</span>'
                f'<span style="font-size:8px; font-weight:700; background:rgba(245,158,11,0.4); color:#fff; border-radius:3px; padding:1px 4px; margin-top:2px;">SPECIAL</span>'
                f'</div>'
            )
        else:
            bg, border, text_col = palette[i % len(palette)]
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
