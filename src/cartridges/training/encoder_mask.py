"""Give the DeepSeek-OCR-2 vision encoder its design attention mask.

The model's Qwen2 encoder (deepencoderv2.py, CustomQwen2ModelInner) must use
a hybrid mask: image tokens attend to all image tokens, query tokens attend
to all image tokens and causally to query tokens. The model code sets this
mask in an override of Qwen2Model._update_causal_mask. transformers >= 4.53
does not call that method: Qwen2Model.forward makes a plain causal mask,
unless the caller gives a mask dict. Thus, without this patch, the encoder
runs with a plain causal mask. vLLM uses the design mask
(docs/hf-vllm-parity.md).

patch_encoder_mask() replaces the forward of the encoder class: it makes the
mask with the model's own _update_causal_mask and gives it as
{"full_attention": mask}, as vLLM does. Class-level, like
fast_decode._patch_moe.
"""

from __future__ import annotations

ENCODER_CLASS = "CustomQwen2ModelInner"


def mask_as_dict_needed() -> bool:
    """True if Qwen2Model.forward ignores _update_causal_mask (>= 4.53)."""
    import transformers
    from packaging.version import Version

    return Version(transformers.__version__) >= Version("4.53.0")


def patch_encoder_mask(model) -> int:
    """Patch each encoder class in `model`. Return the number of classes
    patched now (0 if already patched or not needed)."""
    if not mask_as_dict_needed():
        return 0
    classes = {type(m) for m in model.modules() if type(m).__name__ == ENCODER_CLASS}
    patched = 0
    for cls in classes:
        if getattr(cls, "_design_mask_patch", False):
            continue

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            token_type_ids=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            cache_position=None,
            _cls=cls,
        ):
            self._current_token_type_ids = token_type_ids
            mask = self._update_causal_mask(
                attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
            )
            return super(_cls, self).forward(
                input_ids=input_ids,
                attention_mask={"full_attention": mask},
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )

        cls.forward = forward
        cls._design_mask_patch = True
        patched += 1
    return patched
