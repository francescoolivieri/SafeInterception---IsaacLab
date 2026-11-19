"""M3 safety filter: the learned PNCBF certificate in a CBF-QP on the velocity command."""
from __future__ import annotations

import os

import torch

from target_interception.algorithms.pncbf.constraint_model import INACTIVE_VALUE, PNCBFCertificate
from target_interception.algorithms.pncbf.state import PNCBFObservationModel

from .cbf_qp import qp_stats, solve_cbf_qp, solve_cbf_qp_batched


class PNCBFFilter:
    """Smallest change of the nominal action with dV_i/dt <= -kappa_V (V_i + eps) for every head i.

    kappa_V is alpha_safe when every head is negative (safe) and alpha_unsafe otherwise;
    eps is value_shift. The Lie derivatives use the single-integrator observation dynamics.

    Evaluation takes exact autograd derivatives and solves each QP with quadprog. With
    collection=True (policy-iteration data collection) it uses central finite differences
    and the approximate batched QP on the device instead, which is much faster.
    """

    def __init__(self, obs_layout, *, checkpoint: str, command_scale, alpha_safe: float = 2.0,
                 alpha_unsafe: float = 2.0, value_shift: float = 0.01, collection: bool = False,
                 device=None) -> None:
        if alpha_safe <= 0 or alpha_unsafe <= 0 or value_shift < 0:
            raise ValueError("PNCBF alphas must be positive and value_shift non-negative")
        self.collection = bool(collection)
        self.device = torch.device(device or "cpu")
        self.command_scale = torch.as_tensor(command_scale, dtype=torch.float32, device=self.device).reshape(-1)
        self.alpha_safe = float(alpha_safe)
        self.alpha_unsafe = float(alpha_unsafe)
        self.value_shift = float(value_shift)
        self._last_info: dict = {}
        self.codec = PNCBFObservationModel(obs_layout)
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"PNCBF checkpoint not found: {checkpoint}")

        checkpoint = os.path.abspath(checkpoint)
        data = torch.load(checkpoint, map_location=self.device, weights_only=False)
        model_cfg = data["model_cfg"]
        if data.get("format_version") != 3 or model_cfg.get("kind") != "per_constraint" or model_cfg.get("encoder") != "mlp":
            raise ValueError("unsupported PNCBF checkpoint: expected the per-constraint MLP certificate")
        self._verify_state_spec(data.get("state_spec") or {})
        saved_scale = torch.as_tensor(data.get("command_scale"), dtype=torch.float32, device=self.device).reshape(-1)
        if saved_scale.shape != self.command_scale.shape or not torch.allclose(saved_scale, self.command_scale):
            raise ValueError("PNCBF checkpoint velocity command scale does not match the live environment")
        self.model = PNCBFCertificate(obs_layout, hidden=model_cfg["hidden"], device=self.device)
        self.model.load_checkpoint_state(data["model"])
        self.model.eval()

    def _verify_state_spec(self, saved: dict) -> None:
        live = self.codec.metadata()
        # The six plane heads keep describing the training arena; in the out-of-distribution
        # room the arena differs (as in the thesis evaluation), which is reported, not refused.
        arena = ("arena_min", "arena_max")
        if any(saved.get(key) != live[key] for key in arena):
            print(f"[PNCBF] certificate trained in {saved.get('arena_min')}..{saved.get('arena_max')}, "
                  f"evaluated in {live['arena_min']}..{live['arena_max']}.")
        differences = [key for key in live if key not in arena and saved.get(key) != live[key]]
        if differences:
            details = ", ".join(f"{key}: checkpoint={saved.get(key)!r}, live={live[key]!r}" for key in differences)
            raise ValueError(f"PNCBF checkpoint state spec does not match the live env ({details})")

    def _constraints(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rows G a <= b of every live head. Gradients are needed even under inference mode."""
        if self.collection:
            return self._finite_difference_constraints(obs)
        with torch.inference_mode(False), torch.enable_grad():
            state = self.codec.decode(obs.detach().clone()).detach().requires_grad_(True)
            value = self.model.value(state) + self.value_shift
            # Padding slots are constant inactive rows: skip their gradients.
            heads = [head for head, live in enumerate((value > INACTIVE_VALUE / 2).any(dim=0).tolist()) if live]
            jacobian = torch.zeros(value.shape + state.shape[1:], device=state.device, dtype=state.dtype)
            for position, head in enumerate(heads):
                jacobian[:, head] = torch.autograd.grad(
                    value[:, head].sum(), state, retain_graph=position + 1 < len(heads)
                )[0]
            drift, control = self.codec.drift_and_control(state, self.command_scale)
            lie_f = torch.einsum("bhd,bd->bh", jacobian, drift)
            lie_g = torch.einsum("bhd,bdc->bhc", jacobian, control)
            return self._cbf_rows(value, lie_f, lie_g)

    @torch.no_grad()
    def _finite_difference_constraints(self, obs: torch.Tensor):
        """Central differences along the drift and the three control directions (8 forward passes)."""
        state = self.codec.decode(obs.detach().clone())
        drift, control = self.codec.drift_and_control(state, self.command_scale)
        directions = torch.cat((drift[:, None], control.transpose(1, 2)), dim=1)  # [B, 4, D]
        count, steps, width = directions.shape
        length = directions.norm(dim=-1, keepdim=True)
        unit = directions / length.clamp_min(1e-12)
        epsilon = 1.0e-3  # float32: smaller steps lose more to round-off than they gain
        probes = torch.cat((state[:, None] + epsilon * unit, state[:, None] - epsilon * unit))
        values = self.model.value(probes.reshape(-1, width)).reshape(2, count, steps, -1)
        rates = (values[0] - values[1]) / (2.0 * epsilon) * length  # [B, 4, heads]
        value = self.model.value(state) + self.value_shift
        return self._cbf_rows(value, rates[:, 0], rates[:, 1:].transpose(1, 2))

    def _cbf_rows(self, value, lie_f, lie_g):
        all_safe = (value < 0).all(dim=1, keepdim=True)
        alpha = torch.where(all_safe, value.new_full((value.shape[0], 1), self.alpha_safe),
                            value.new_full((value.shape[0], 1), self.alpha_unsafe))
        bound = -(lie_f + alpha * value)
        return lie_g.detach(), bound.detach(), value.detach()

    def __call__(self, obs: torch.Tensor, nominal_action: torch.Tensor) -> torch.Tensor:
        action = nominal_action.to(self.device)
        G, b, value = self._constraints(obs.to(self.device))
        solve = solve_cbf_qp_batched if self.collection else solve_cbf_qp
        safe_xyz, slack = solve(action[:, :3], G, b)
        safe = action.clone()
        safe[:, :3] = safe_xyz
        self._last_info = qp_stats(action[:, :3], safe_xyz, G, b, slack)
        self._last_info.update({
            "certificate_max_mean": float(value.amax(dim=1).mean().item()),
            "certificate_max": float(value.max().item()),
            "certificate_safe_fraction": float((value < 0).all(dim=1).float().mean().item()),
        })
        return safe.to(nominal_action.device)

    def info(self) -> dict:
        return dict(self._last_info)
