"""Verify: HF Qwen2 encoder with the intended mask (passed as a mask dict, as
vLLM does) matches vLLM's saved encoder outputs. Same merged weights.

Run scripts/parity/stage_parity.py first (it saves the vLLM tensors), then:
uv run modal run scripts/parity/mask_verify.py --pages 63,52,44
"""

import modal

from cartridges.training.finetune import image as _training_image

SHA = "a2f23f486707d22fb735e48c9cd258a6660e57ee"
app = modal.App("cartridges-mask-verify")
vols = {"/cache": modal.Volume.from_name("cartridges-training-cache"),
        "/parity": modal.Volume.from_name("cartridges-parity-scratch")}

@app.function(image=_training_image.add_local_python_source("cartridges"), gpu="A100", timeout=20 * 60, volumes=vols)
def run(pngs: dict[int, bytes]) -> dict:
    import io
    import sys
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoTokenizer
    sys.path.insert(0, "/cache")
    from cartridges.training.collator import DeepSeekOCRDataCollator, to_conversation

    path = f"/cache/models/{SHA}"
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModel.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                      attn_implementation="sdpa", use_safetensors=True).cuda().eval()
    inner = model.model
    collator = DeepSeekOCRDataCollator(tokenizer=tok, model=model)
    enc = inner.qwen2_model.model.model  # CustomQwen2ModelInner
    cls = type(enc)
    original = cls.forward

    def fixed(self, *args, attention_mask=None, inputs_embeds=None, token_type_ids=None, **kw):
        self._current_token_type_ids = token_type_ids
        mask = {"full_attention": self._update_causal_mask(attention_mask, inputs_embeds, None, None, False)}
        return super(cls, self).forward(*args, attention_mask=mask, inputs_embeds=inputs_embeds, **kw)

    store = {}
    view = {}
    inner.sam_model.register_forward_pre_hook(lambda m, a: view.__setitem__("v", "global" if a[0].shape[-1] == 1024 else "local"))
    for name, key in (("qwen2_model", "qwen2"), ("projector", "proj")):
        getattr(inner, name).register_forward_hook(
            lambda m, a, out, key=key: store.__setitem__(f"{key}/{view['v']}", out.detach().to(torch.bfloat16).cpu()))

    def stats(a, b):
        a, b = a.float().flatten(), b.float().flatten()
        return {"rel_l2": round(((a - b).norm() / a.norm()).item(), 5),
                "cos": round(torch.nn.functional.cosine_similarity(a, b, dim=0).item(), 6)}

    result = {}
    for page, data in pngs.items():
        v = torch.load(f"/parity/vllm_{page}.pt")
        t = collator.process_single_sample(to_conversation({"image": Image.open(io.BytesIO(data)).convert("RGB"), "text": ""})["messages"])
        n = t["prompt_token_count"]
        for mode, fwd in (("as_shipped", original), ("intended_mask", fixed)):
            cls.forward = fwd
            store.clear()
            with torch.no_grad():
                model(input_ids=t["input_ids"][:n].unsqueeze(0).cuda(),
                      images=[(t["images_crop"].cuda(), t["images_ori"].cuda())],
                      images_seq_mask=t["images_seq_mask"][:n].unsqueeze(0).cuda(),
                      images_spatial_crop=t["images_spatial_crop"], use_cache=False)
            result[f"{page}/{mode}"] = {k: stats(store[k], v[k]) for k in ("qwen2/global", "qwen2/local", "proj/global", "proj/local") if k in store}
    cls.forward = original
    return result

@app.local_entrypoint()
def main(pages: str = "63,52,44"):
    import io
    import json
    from datasets import load_from_disk
    test = load_from_disk("data/train/hf_dataset")["test"]
    pngs = {}
    for i in map(int, pages.split(",")):
        buf = io.BytesIO()
        test[i]["image"].save(buf, format="PNG")
        pngs[i] = buf.getvalue()
    print(json.dumps(run.remote(pngs), indent=1))
