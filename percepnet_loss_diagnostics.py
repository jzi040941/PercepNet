"""
PercepNet loss & gradient diagnostics utilities.

Save as percepnet_loss_diagnostics.py into your training repository and import:
from percepnet_loss_diagnostics import CustomLossStable, run_diagnostics_step, run_ablation_suite

Purpose:
- Provide a numerically-stable variant of the CustomLoss used in PercepNet training.
- Run a single diagnostic step (forward + backward) and report loss components and gradient statistics.
- Convenience function to run a small ablation sweep.

Usage (example, run from your training script):
    loss_fn = CustomLossStable(gamma=0.5, C4=10.0, eps=0.0, clamp_preds=False)
    diagnostics, param_grad_norms = run_diagnostics_step(model, optimizer, loss_fn, inputs, targets, "baseline", device=device)
    # For ablations:
    results = run_ablation_suite(model, optimizer, inputs, targets, device=device)

Notes:
- This code does not change model weights unless you call optimizer.step() after calling run_diagnostics_step.
- The functions are robust for CPU/GPU tensors.
"""

import torch
import math
from typing import Tuple, Dict, List, Any

# Adjust N_BANDS if your repo uses a different number of ERB bands
N_BANDS = 34
OUT_DIM = N_BANDS * 2  # gb + rb -> 68


class CustomLossStable(torch.nn.Module):
    """
    Stable / diagnostic variant of the PercepNet loss used in the repo.
    Options:
      - gamma: compression exponent (paper uses 0.5)
      - C4: multiplier for the quartic penalty on gain errors
      - eps: small positive value used to avoid fractional-power instabilities:
             if eps > 0, values are clamped to [eps, 1-eps] before pow.
      - clamp_preds: if True, clamp both predictions and targets to [eps, 1-eps]
                     (useful when eps > 0).
    The forward() can return full components when return_components=True.
    """
    def __init__(self, gamma: float = 0.5, C4: float = 10.0, eps: float = 0.0, clamp_preds: bool = False):
        super(CustomLossStable, self).__init__()
        assert gamma > 0.0, "gamma must be > 0"
        self.gamma = float(gamma)
        self.C4 = float(C4)
        self.eps = float(eps)
        self.clamp_preds = bool(clamp_preds)

    def _safe_pow(self, x: torch.Tensor) -> torch.Tensor:
        # clamp to [eps, 1-eps] if requested; otherwise clamp only from below by eps
        if self.clamp_preds:
            if self.eps > 0.0:
                x = torch.clamp(x, min=self.eps, max=1.0 - self.eps)
            else:
                x = torch.clamp(x, min=0.0, max=1.0)  # ensures no >1 values explode
        else:
            if self.eps > 0.0:
                x = torch.clamp(x, min=self.eps)
        # return x ** gamma (torch.pow handles tensors)
        return torch.pow(x, self.gamma)

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, return_components: bool = False) -> Any:
        """
        outputs: [B, T, OUT_DIM]
        targets: [B, T, OUT_DIM]
        """
        if outputs.dim() != 3 or targets.dim() != 3:
            raise ValueError("outputs and targets must be 3-D tensors [B, T, OUT_DIM]")

        if outputs.size(-1) != OUT_DIM or targets.size(-1) != OUT_DIM:
            raise ValueError(f"expected last-dim == {OUT_DIM} (gb+rb). Got {outputs.size(-1)} and {targets.size(-1)}")

        gb_hat = outputs[..., :N_BANDS]
        rb_hat = outputs[..., N_BANDS:OUT_DIM]
        gb = targets[..., :N_BANDS]
        rb = targets[..., N_BANDS:OUT_DIM]

        # Compute compressed values with optional eps/clamping
        gb_c = self._safe_pow(gb)
        gb_hat_c = self._safe_pow(gb_hat)

        one_minus_rb_c = self._safe_pow(1.0 - rb)
        one_minus_rb_hat_c = self._safe_pow(1.0 - rb_hat)

        # per-paper losses (mean over all elements)
        Lg = torch.mean((gb_c - gb_hat_c) ** 2)
        Lq = torch.mean((gb_c - gb_hat_c) ** 4)
        Lr = torch.mean((one_minus_rb_c - one_minus_rb_hat_c) ** 2)

        total = Lg + self.C4 * Lq + Lr

        if return_components:
            return total, Lg.detach(), Lq.detach(), Lr.detach()
        return total


def _collect_grad_stats(model: torch.nn.Module) -> Tuple[float, float, float, int, int, int, List[Tuple[str, float]]]:
    """
    Inspect model parameter gradients after backward() has been called.

    Returns:
      - grad_max: maximum absolute gradient element across all params
      - total_grad_norm: global L2 norm across flattened gradients
      - total_params_with_grad: number of parameters that had non-None gradients
      - count_gt_1: number of gradient elements with |g| > 1
      - count_gt_10: number of gradient elements with |g| > 10
      - total_grad_count: total number of gradient elements considered
      - top_param_grad_norms: list[(param_name, l2_norm)] sorted descending by norm (top 12)
    """
    grad_max = 0.0
    total_grad_count = 0
    count_gt_1 = 0
    count_gt_10 = 0
    param_grad_norms: List[Tuple[str, float]] = []
    sum_sq = 0.0
    params_with_grad = 0

    for name, p in model.named_parameters():
        g = p.grad
        if g is None:
            continue
        params_with_grad += 1
        g_det = g.detach()
        abs_g = g_det.abs()
        local_max = float(abs_g.max().cpu().item())
        grad_max = max(grad_max, local_max)

        # counts
        count_gt_1 += int((abs_g > 1.0).sum().cpu().item())
        count_gt_10 += int((abs_g > 10.0).sum().cpu().item())
        total_grad_count += g_det.numel()

        # l2 norm of this parameter
        l2 = float(g_det.view(-1).norm(2).cpu().item())
        param_grad_norms.append((name, l2))

        sum_sq += float((g_det.view(-1).double() ** 2).sum().cpu().item())

    total_grad_norm = float(math.sqrt(sum_sq)) if sum_sq > 0.0 else 0.0

    # sort top norms
    top_param_grad_norms = sorted(param_grad_norms, key=lambda x: x[1], reverse=True)[:12]

    return grad_max, total_grad_norm, params_with_grad, count_gt_1, count_gt_10, total_grad_count, top_param_grad_norms


def run_diagnostics_step(model: torch.nn.Module,
                         optimizer: torch.optim.Optimizer,
                         loss_fn: CustomLossStable,
                         inputs: torch.Tensor,
                         targets: torch.Tensor,
                         label: str = "diag",
                         device: str = "cpu",
                         do_backward: bool = True,
                         keep_graph: bool = False) -> Tuple[Dict[str, Any], List[Tuple[str, float]]]:
    """
    Run one forward/backward diagnostic step and print diagnostics.

    Returns:
      diagnostics: dict with the requested metrics
      param_grad_norms: list of (param_name, l2_norm) collected (top norms first)
    """
    model.to(device)
    inputs = inputs.to(device)
    targets = targets.to(device)

    model.train()  # we want training-mode behavior for gradients etc.
    optimizer.zero_grad()

    outputs = model(inputs)  # assume model accepts inputs directly: [B,T,x_dim] -> [B,T,OUT_DIM]
    if not isinstance(outputs, torch.Tensor):
        # if your model returns (outputs, other), adjust as needed
        outputs = outputs[0]

    # clamp outputs and targets to expected range for diagnostics printing
    out_min = float(outputs.min().cpu().item())
    out_max = float(outputs.max().cpu().item())

    total, Lg, Lq, Lr = loss_fn(outputs, targets, return_components=True)

    if do_backward:
        # backward to populate gradients (no optimizer.step here)
        total.backward(retain_graph=keep_graph)

    # collect gradient statistics
    grad_max, total_grad_norm, params_with_grad, count_gt_1, count_gt_10, total_grad_count, top_param_grad_norms = _collect_grad_stats(model)

    prop_large_gt_1 = (count_gt_1 / total_grad_count) if total_grad_count > 0 else 0.0
    prop_large_gt_10 = (count_gt_10 / total_grad_count) if total_grad_count > 0 else 0.0

    diagnostics = {
        "label": label,
        "loss_total": float(total.cpu().item()),
        "Lg": float(Lg),
        "Lq": float(Lq),
        "Lr": float(Lr),
        "outputs_min": out_min,
        "outputs_max": out_max,
        "grad_max": grad_max,
        "prop_large_gt_1": prop_large_gt_1,
        "prop_large_gt_10": prop_large_gt_10,
        "total_grad_norm": total_grad_norm,
        "params_with_grad": params_with_grad,
        "total_grad_count": total_grad_count,
    }

    # Print a compact diagnostic summary
    print(f"[DIAG {label}] loss_total={diagnostics['loss_total']:.6e}  Lg={diagnostics['Lg']:.6e}  Lq={diagnostics['Lq']:.6e}  Lr={diagnostics['Lr']:.6e}")
    print(f"[DIAG {label}] outputs_min={out_min:.6e}  outputs_max={out_max:.6e}")
    print(f"[DIAG {label}] grad_max={grad_max:.6e}  prop>|g|>1={prop_large_gt_1:.4f}  prop>|g|>10={prop_large_gt_10:.4f}  total_grad_norm={total_grad_norm:.6e}")
    print(f"[DIAG {label}] params_with_grad={params_with_grad}  total_grad_count={total_grad_count}")
    print(f"[DIAG {label}] top parameter grad norms (name, l2):")
    for name, nrm in top_param_grad_norms:
        print(f"    {name:40s}  {nrm:.6e}")

    return diagnostics, top_param_grad_norms


def run_ablation_suite(model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       inputs: torch.Tensor,
                       targets: torch.Tensor,
                       device: str = "cpu",
                       gammas=(1.0, 0.75, 0.5),
                       C4s=(0.0, 1.0, 10.0),
                       eps_values=(0.0, 1e-8),
                       clamp_options=(False, True)) -> Dict[str, Dict[str, Any]]:
    """
    Run a small grid search over gamma, C4, eps, clamp_preds combinations and record diagnostics.

    Returns a nested dict keyed by a concise variant name -> diagnostics dict.
    """
    results = {}
    baseline_loss_fn = CustomLossStable(gamma=0.5, C4=10.0, eps=0.0, clamp_preds=False)

    # baseline first
    d, top = run_diagnostics_step(model, optimizer, baseline_loss_fn, inputs, targets, label="baseline", device=device)
    results["baseline"] = d

    # sweep
    for gamma in gammas:
        for C4 in C4s:
            for eps in eps_values:
                for clamp in clamp_options:
                    name = f"g{gamma}_C4{int(C4)}_eps{eps}_clamp{int(clamp)}"
                    loss_fn = CustomLossStable(gamma=gamma, C4=C4, eps=eps, clamp_preds=clamp)
                    d, top = run_diagnostics_step(model, optimizer, loss_fn, inputs, targets, label=name, device=device)
                    results[name] = d

    return results
#

if __name__ == "__main__":
    # Minimal smoke test / example usage when running the script standalone.
    # NOTE: replace ModelStub with your real model and shapes.
    class ModelStub(torch.nn.Module):
        def __init__(self, in_dim=70):
            super().__init__()
            self.lin = torch.nn.Linear(in_dim, OUT_DIM)

        def forward(self, x):
            # expect x: [B, T, in_dim] -> produce [B, T, OUT_DIM]
            B, T, _ = x.shape
            y = self.lin(x.view(B * T, -1))
            return y.view(B, T, OUT_DIM)

    # Quick smoke example (does not run full training)
    B = 4
    T = 64
    in_dim = 70
    device = "cpu"
    model = ModelStub(in_dim=in_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # fake inputs/targets representative shapes; targets must be in [0,1]
    inputs = torch.randn(B, T, in_dim, dtype=torch.float32)
    targets = torch.rand(B, T, OUT_DIM, dtype=torch.float32)

    loss_fn = CustomLossStable(gamma=0.5, C4=10.0, eps=0.0, clamp_preds=False)
    run_diagnostics_step(model, optimizer, loss_fn, inputs, targets, label="smoke-test", device=device)
