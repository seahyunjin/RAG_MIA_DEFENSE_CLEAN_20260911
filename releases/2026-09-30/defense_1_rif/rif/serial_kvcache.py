"""Gold SERIAL_KVCACHE reference decoder for frozen old CF-Clip."""

from __future__ import annotations

from .cfclip import GAMMA, cf_clip_logits


class SerialKVCacheCFClip:
    def __init__(self, model, tokenizer, gamma: float = GAMMA, repetition_penalty: float = 1.05):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.gamma = gamma
        self.penalty = repetition_penalty
        self.body = model.model
        self.head = model.lm_head

    def _prefill(self, ids):
        import torch

        x = torch.as_tensor(ids, dtype=torch.long, device=self.model.device).view(1, -1)
        mask = torch.ones_like(x)
        pos = torch.arange(x.shape[1], device=x.device).view(1, -1)
        out = self.body(input_ids=x, attention_mask=mask, position_ids=pos, use_cache=True)
        return {
            "hidden": out.last_hidden_state[:, -1, :],
            "past": out.past_key_values,
            "mask": mask,
            "position": x.shape[1],
        }

    def _advance(self, state, token):
        import torch

        x = torch.as_tensor([token], dtype=torch.long, device=self.model.device).view(1, 1)
        mask = torch.cat(
            (state["mask"], torch.ones((1, 1), dtype=state["mask"].dtype, device=x.device)),
            -1,
        )
        pos = torch.as_tensor([[state["position"]]], device=x.device)
        out = self.body(
            input_ids=x,
            attention_mask=mask,
            position_ids=pos,
            past_key_values=state["past"],
            use_cache=True,
        )
        state.update(
            hidden=out.last_hidden_state[:, -1, :],
            past=out.past_key_values,
            mask=mask,
            position=state["position"] + 1,
        )

    def _penalize(self, scores, seen):
        import torch

        if not seen:
            return scores
        idx = torch.unique(torch.as_tensor(seen, dtype=torch.long, device=scores.device))
        values = scores.index_select(0, idx)
        changed = torch.where(values < 0, values * self.penalty, values / self.penalty)
        return scores.clone().scatter(0, idx, changed)

    def decode(self, branch_prompt_ids, max_new_tokens: int = 64):
        """Decode from FULL then DROP-1..DROP-k prompts serially and exactly."""
        import torch

        if len(branch_prompt_ids) < 2:
            raise ValueError("Need one FULL and at least one DROP prompt")
        eos_ids = self.model.generation_config.eos_token_id
        eos = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
        states = [self._prefill(ids) for ids in branch_prompt_ids]
        generated = []
        trace = []
        with torch.inference_mode():
            for position in range(max_new_tokens):
                logits = torch.cat([self.head(state["hidden"]).float() for state in states], 0)
                logp = torch.log_softmax(logits, dim=-1)
                defended, max_lift, clip = cf_clip_logits(
                    logits[0], logp[0], logp[1:], self.gamma
                )
                token = int(
                    torch.argmax(
                        self._penalize(defended, branch_prompt_ids[0] + generated)
                    ).item()
                )
                trace.append(
                    {
                        "position": position,
                        "token_id": token,
                        "g_max": float(max_lift[token].item()),
                        "chosen_clip": float(clip[token].item()),
                        "clip_decision": bool(clip[token].item() > 0),
                    }
                )
                generated.append(token)
                if token in eos:
                    break
                for state in states:
                    self._advance(state, token)
        answer_ids = generated[:-1] if generated and generated[-1] in eos else generated
        return {
            "token_ids": answer_ids,
            "text": self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip(),
            "trace": trace[: len(answer_ids)],
        }

