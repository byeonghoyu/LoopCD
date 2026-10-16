from __future__ import annotations
import math
import types

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast


def _cd_blend(log_p_e: torch.Tensor, log_p_a: torch.Tensor,
              lambda_cd: float, alpha_plaus: float) -> torch.Tensor:
    log_alpha = math.log(alpha_plaus)
    max_log_p_e = log_p_e.max(dim=-1, keepdim=True).values
    plausible = log_p_e >= (max_log_p_e + log_alpha)
    cd = log_p_e - lambda_cd * log_p_a
    return torch.where(plausible, cd, torch.full_like(cd, float("-inf")))


def _cd_forward_ouro(self, *,
                     input_ids=None,
                     attention_mask=None,
                     position_ids=None,
                     past_key_values=None,
                     inputs_embeds=None,
                     labels=None,
                     use_cache=None,
                     cache_position=None,
                     logits_to_keep=0,
                     use_weighted_exit=False,
                     exit_at_step=None,
                     exit_threshold=None,
                     **kwargs):
    cfg = self._cd_config
    k_a   = int(cfg["k_amateur"])
    lam   = float(cfg["lambda_cd"])
    alpha = float(cfg["alpha_plaus"])

    outputs, hidden_states_list, gate_list = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        cache_position=cache_position,
        **kwargs,
    )

    n_steps = len(hidden_states_list)
    if k_a < 0 or k_a >= n_steps:
        raise ValueError(f"k_amateur={k_a} out of range [0,{n_steps})")
    if k_a == n_steps - 1:
        slice_indices = (slice(-logits_to_keep, None)
                         if isinstance(logits_to_keep, int) else logits_to_keep)
        h_expert = hidden_states_list[-1]
        new_logits = self.lm_head(h_expert[:, slice_indices, ...]
                                  if isinstance(slice_indices, slice)
                                  else h_expert.index_select(1, slice_indices.to(h_expert.device)))
        return CausalLMOutputWithPast(loss=None, logits=new_logits,
                                      past_key_values=outputs.past_key_values)

    h_amateur = hidden_states_list[k_a]
    h_expert  = hidden_states_list[-1]

    slice_indices = (slice(-logits_to_keep, None)
                     if isinstance(logits_to_keep, int) else logits_to_keep)
    def _select(t):
        if isinstance(slice_indices, slice):
            return t[:, slice_indices, ...]
        return t.index_select(1, slice_indices.to(t.device))

    logits_e_slice = self.lm_head(_select(h_expert))
    logits_a_slice = self.lm_head(_select(h_amateur))

    log_p_e_last = F.log_softmax(logits_e_slice[:, -1, :].float(), dim=-1)
    log_p_a_last = F.log_softmax(logits_a_slice[:, -1, :].float(), dim=-1)
    cd_last = _cd_blend(log_p_e_last, log_p_a_last, lam, alpha)

    new_logits = logits_e_slice.clone()
    new_logits[:, -1, :] = cd_last.to(new_logits.dtype)

    return CausalLMOutputWithPast(
        loss=None,
        logits=new_logits,
        past_key_values=outputs.past_key_values,
    )


def install_ouro_cd(model, k_amateur: int = 1, lambda_cd: float = 0.3,
                    alpha_plaus: float = 0.1) -> None:
    if hasattr(model, "_original_forward"):
        return
    model._original_forward = model.forward
    model._cd_config = {
        "k_amateur":   k_amateur,
        "lambda_cd":   lambda_cd,
        "alpha_plaus": alpha_plaus,
    }
    model.forward = types.MethodType(_cd_forward_ouro, model)


def uninstall_ouro_cd(model) -> None:
    if hasattr(model, "_original_forward"):
        model.forward = model._original_forward
        del model._original_forward
        del model._cd_config
