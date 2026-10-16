from __future__ import annotations
import math
import types

import torch
import torch.nn.functional as F


def _cd_blend(log_p_e: torch.Tensor, log_p_a: torch.Tensor,
              lambda_cd: float, alpha_plaus: float) -> torch.Tensor:
    log_alpha = math.log(alpha_plaus)
    max_log_p_e = log_p_e.max(dim=-1, keepdim=True).values
    plausible = log_p_e >= (max_log_p_e + log_alpha)
    cd = log_p_e - lambda_cd * log_p_a
    return torch.where(plausible, cd, torch.full_like(cd, float("-inf")))


def _cd_forward(self, *,
                input_ids: torch.Tensor | None = None,
                input_embeds: torch.Tensor | None = None,
                input_states=None,
                attention_mask=None,
                position_ids=None,
                cache_position=None,
                past_key_values=None,
                num_steps=None,
                init_scale: float = 1.0,
                use_cache: bool = False,
                output_details: dict | None = None,
                labels=None,
                **kwargs):
    cfg = self._cd_config
    k_a   = int(cfg["k_amateur"])
    lam   = float(cfg["lambda_cd"])
    alpha = float(cfg["alpha_plaus"])

    if position_ids is None and cache_position is None:
        freqs_cis = self.freqs_cis[:, : input_ids.shape[1]]
    elif position_ids is not None:
        freqs_cis = self.freqs_cis.index_select(1, position_ids.squeeze())
    else:
        freqs_cis = self.freqs_cis[:, cache_position]

    if input_embeds is None:
        input_embeds = self.transformer.wte(input_ids)
    if self.emb_scale != 1:
        input_embeds = input_embeds * self.emb_scale

    if use_cache and past_key_values is None:
        import sys as _sys
        HDCache = _sys.modules[type(self).__module__].HuginnDynamicCache
        past_key_values = HDCache()

    block_idx = torch.tensor(-1, device=torch.device("cpu"), dtype=torch.long)
    for block in self.transformer.prelude:
        block_idx += 1
        input_embeds = block(input_embeds, freqs_cis, block_idx, None, past_key_values)

    r = int(num_steps) if num_steps is not None else 32
    x = self.initialize_state(input_embeds, scale=init_scale) if input_states is None else input_states.clone()
    h_amateur = None
    for step in range(r):
        x, block_idx = self.core_block_forward(
            x, input_embeds, freqs_cis, None, past_key_values, block_idx, current_step=step,
        )
        if (step + 1) == k_a:
            h_amateur = x.clone()
    h_expert = x
    if h_amateur is None:
        h_amateur = h_expert

    amateur_out = self.predict_from_latents(
        h_amateur, attention_mask=None, position_ids=position_ids,
        cache_position=cache_position, past_key_values=past_key_values)
    expert_out = self.predict_from_latents(
        h_expert, attention_mask=None, position_ids=position_ids,
        cache_position=cache_position, past_key_values=past_key_values)

    log_p_a_last = F.log_softmax(amateur_out.logits[:, -1, :].float(), dim=-1)
    log_p_e_last = F.log_softmax(expert_out.logits[:, -1, :].float(), dim=-1)
    cd_last = _cd_blend(log_p_e_last, log_p_a_last, lam, alpha)

    new_logits = expert_out.logits.clone()
    new_logits[:, -1, :] = cd_last.to(new_logits.dtype)

    from transformers import GenerationMixin
    OutCls = type(expert_out)
    return OutCls(
        loss=torch.as_tensor(0.0),
        log_ppl=torch.as_tensor(0.0),
        logits=new_logits,
        past_key_values=expert_out.past_key_values,
        latent_states=h_expert,
    )


def install(model, k_amateur: int = 8, lambda_cd: float = 0.3,
            alpha_plaus: float = 0.1) -> None:
    if hasattr(model, "_original_forward"):
        return
    model._original_forward = model.forward
    model._cd_config = {
        "k_amateur":   k_amateur,
        "lambda_cd":   lambda_cd,
        "alpha_plaus": alpha_plaus,
    }
    model.forward = types.MethodType(_cd_forward, model)


def uninstall(model) -> None:
    if hasattr(model, "_original_forward"):
        model.forward = model._original_forward
        del model._original_forward
        del model._cd_config
