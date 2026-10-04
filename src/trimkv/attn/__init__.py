import torch

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from . import eager_attn
from . import flex_attn

TRIMKV_ATTENTION_IMPLEMENTATIONS = {
    "rg_attn_eager": eager_attn.retention_gated_attention_forward, # Retention-Gated Attention implementation using Eager Attention
    "rg_attn_flex": flex_attn.retention_gated_attention_forward, # Retention-Gated Attention implementation using Flex Attention
    "attn_eager": eager_attn.eager_attention_forward, # Standard Attention implementation using Eager Attention
}


def get_trimkv_wrapper(attn_impl: str):
    attn_fn = ALL_ATTENTION_FUNCTIONS[attn_impl]

    def attn_wrapper(*args, **kwargs):
        kwargs.pop("retention_weights", None)
        kwargs.pop("kv_positions", None)
        kwargs.pop("rg_dropout", None)
        kwargs.pop("flash_attn_kwargs", None)
        attn_output, attn_weights = attn_fn(*args, **kwargs)
        return attn_output, attn_weights, None
    
    return attn_wrapper

def get_attention_interface(attn_impl: str, compile=False):
    if attn_impl in ("db_flash_attention_2", "paged_flash_attention_2"):
        # FlexAttention training does not need the optional flash-attn package.
        try:
            from . import db_flash_attn
        except ModuleNotFoundError as exc:
            if exc.name == "flash_attn" or (exc.name or "").startswith("flash_attn."):
                raise ImportError(
                    f"{attn_impl} requires the optional flash-attn package. "
                    "Use rg_attn_flex for TrimKV training without flash-attn."
                ) from exc
            raise
        attention_inference = (
            db_flash_attn.dynamic_kv_budget_attention_forward
            if attn_impl == "db_flash_attention_2"
            else db_flash_attn.paged_flash_attention_forward
        )
    elif attn_impl not in TRIMKV_ATTENTION_IMPLEMENTATIONS:
        attention_inference = get_trimkv_wrapper(attn_impl)
    else:
        attention_inference = TRIMKV_ATTENTION_IMPLEMENTATIONS.get(attn_impl, None)

    if compile:
        attention_inference = torch.compile(attention_inference)

    return attention_inference
