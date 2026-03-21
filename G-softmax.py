import torch
import torch.nn.functional as F
import numpy as np

# ============================================================
# Your 5 classes — exactly as in your dual network
# ============================================================
classes = torch.tensor([-0.0001, -0.005, -0.01, -0.05, -10.0], dtype=torch.float64)
class_labels = ["-0.0001", "-0.005", "-0.01", "-0.05", "-10.0"]

print("=" * 60)
print("CLASSES (negated generator costs):")
for i, (c, l) in enumerate(zip(classes, class_labels)):
    print(f"  Class {i}: {l}")

# ============================================================
# Simulate network logits for one node
# Class 1 (-0.005) and Class 2 (-0.01) are close — hard to distinguish
# ============================================================
logits = torch.tensor([0.5, 2.1, 1.9, 0.2, -1.0], dtype=torch.float64, requires_grad=True)

print("\n" + "=" * 60)
print("LOGITS (network's raw confidence per class):")
for i, (l, lbl) in enumerate(zip(logits, class_labels)):
    print(f"  Class {i} ({lbl}): {l.item():.2f}")

# ============================================================
# METHOD 1: Standard softmax (what you currently do during training)
# ============================================================
print("\n" + "=" * 60)
print("METHOD 1 — Standard softmax (current training behaviour)")
print("-" * 60)

probs = F.softmax(logits, dim=-1)
soft_lambda = (probs * classes).sum()

print("Probabilities:")
for i, (p, lbl) in enumerate(zip(probs, class_labels)):
    bar = "#" * int(p.item() * 40)
    print(f"  Class {i} ({lbl}): {p.item():.4f}  {bar}")

print(f"\nOutput lambda = {soft_lambda.item():.6f}")
print(f"  --> NOT a real class value. No generator costs this much.")
print(f"  --> This is what the network trains on — a fiction.")

# ============================================================
# METHOD 2: Argmax (what you do at inference)
# ============================================================
print("\n" + "=" * 60)
print("METHOD 2 — Argmax (current inference behaviour)")
print("-" * 60)

chosen_idx = logits.argmax().item()
hard_lambda = classes[chosen_idx]

print(f"Chosen class: {chosen_idx} ({class_labels[chosen_idx]})")
print(f"Output lambda = {hard_lambda.item():.6f}")
print(f"  --> This IS a real class value.")
print(f"  --> But gradient = 0. Cannot backpropagate through argmax.")

# ============================================================
# METHOD 3: Gumbel-softmax hard=True (straight-through estimator)
# ============================================================
print("\n" + "=" * 60)
print("METHOD 3 — Gumbel-softmax hard=True")
print("-" * 60)

torch.manual_seed(42)
y_hard = F.gumbel_softmax(logits, tau=1.0, hard=True)

chosen_idx_g = y_hard.argmax().item()
hard_g_lambda = (y_hard * classes).sum()

print("One-hot output (hard forward pass):")
for i, (h, lbl) in enumerate(zip(y_hard, class_labels)):
    marker = " <-- CHOSEN" if i == chosen_idx_g else ""
    print(f"  Class {i} ({lbl}): {h.item():.0f}{marker}")

print(f"\nOutput lambda = {hard_g_lambda.item():.6f}")
print(f"  --> This IS a real class value (same as argmax).")

# Backpropagate
hard_g_lambda.backward()
print(f"\nGradient on logits: {logits.grad.tolist()}")
print(f"  --> Non-zero! Backprop works through the soft distribution.")

# ============================================================
# THE STRAIGHT-THROUGH TRICK — show exactly what happens
# ============================================================
print("\n" + "=" * 60)
print("STRAIGHT-THROUGH TRICK — under the hood")
print("-" * 60)

logits2 = torch.tensor([0.5, 2.1, 1.9, 0.2, -1.0], dtype=torch.float64, requires_grad=True)
tau = 1.0

# Step 1: add Gumbel noise
gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits2) + 1e-8) + 1e-8)
noisy_logits = logits2 + gumbel_noise

# Step 2: soft distribution with temperature
y_soft = F.softmax(noisy_logits / tau, dim=-1)

# Step 3: hard one-hot from argmax of soft
idx = y_soft.argmax(dim=-1)
y_hard_manual = F.one_hot(idx, num_classes=5).double()

# Step 4: straight-through — forward uses hard, backward uses soft
# y_hard_manual has no gradient. y_soft does.
# The trick: subtract soft (detached) then add soft (with grad)
st_output = y_hard_manual - y_soft.detach() + y_soft

print("y_soft (used for gradients):")
for i, (s, lbl) in enumerate(zip(y_soft, class_labels)):
    print(f"  Class {i} ({lbl}): {s.item():.4f}")

print(f"\ny_hard_manual (used for forward value):")
for i, (h, lbl) in enumerate(zip(y_hard_manual, class_labels)):
    marker = " <-- CHOSEN" if h.item() == 1.0 else ""
    print(f"  Class {i} ({lbl}): {h.item():.0f}{marker}")

print(f"\nst_output (= y_hard_manual - y_soft.detach() + y_soft):")
for i, (s, lbl) in enumerate(zip(st_output, class_labels)):
    print(f"  Class {i} ({lbl}): {s.item():.4f}")

print(f"\n  Value is identical to y_hard (real class).")
print(f"  But computationally connected to y_soft — gradients flow.")

lambda_st = (st_output * classes).sum()
lambda_st.backward()
print(f"\nGradient on logits: {logits2.grad.tolist()}")
print(f"  --> Non-zero. Network can learn from this.")

# ============================================================
# TEMPERATURE EFFECT — show how tau changes sharpness
# ============================================================
print("\n" + "=" * 60)
print("TEMPERATURE EFFECT — different tau values")
print("-" * 60)

logits3 = torch.tensor([0.5, 2.1, 1.9, 0.2, -1.0], dtype=torch.float64)
torch.manual_seed(42)
noise = -torch.log(-torch.log(torch.rand_like(logits3) + 1e-8) + 1e-8)
noisy = logits3 + noise

for tau in [5.0, 1.0, 0.5, 0.1]:
    probs_t = F.softmax(noisy / tau, dim=-1)
    chosen = probs_t.argmax().item()
    print(f"\n  tau={tau:.1f}: probs = {[f'{p:.3f}' for p in probs_t.tolist()]}")
    print(f"           chosen class = {chosen} ({class_labels[chosen]})")

print("\n  High tau → flat distribution → more random")
print("  Low tau  → sharp distribution → closer to argmax")
print("  Anneal from high → low during training")

# ============================================================
# STOCHASTICITY — sample 1000 times, see class frequencies
# ============================================================
print("\n" + "=" * 60)
print("STOCHASTICITY — sample 1000 times with tau=1.0")
print("-" * 60)

logits4 = torch.tensor([0.5, 2.1, 1.9, 0.2, -1.0], dtype=torch.float64)
counts = [0] * 5
N = 1000
for _ in range(N):
    y = F.gumbel_softmax(logits4, tau=1.0, hard=True)
    counts[y.argmax().item()] += 1

print("Class frequencies (hard=True, tau=1.0):")
for i, (c, lbl) in enumerate(zip(counts, class_labels)):
    bar = "#" * (c // 20)
    print(f"  Class {i} ({lbl}): {c:4d}/1000  {bar}")

print(f"\n  Classes 1 and 2 are both chosen frequently because")
print(f"  their logits (2.1 and 1.9) are close.")
print(f"  Gumbel noise sometimes pushes one above the other.")
print(f"  The network learns from BOTH outcomes — this is the key advantage.")

# ============================================================
# WHY THIS MATTERS FOR YOUR DUAL NETWORK
# ============================================================
print("\n" + "=" * 60)
print("WHY THIS MATTERS FOR YOUR DUAL NETWORK")
print("-" * 60)
print("""
  Current training (softmax):
    Logits [2.1, 1.9] for classes [-0.005, -0.01]
    Output: 0.55*(-0.005) + 0.45*(-0.01) = -0.00725
    --> Network never sees what happens when it picks -0.005 OR -0.01
    --> It trains on -0.00725 which is never a real shadow price

  With Gumbel-softmax hard=True:
    Sometimes picks -0.005 (class 1 wins the noise lottery)
    Sometimes picks -0.01 (class 2 wins the noise lottery)
    --> Network trains on REAL shadow prices
    --> Learns: "when I pick -0.005, the dual objective is X"
    --> Learns: "when I pick -0.01, the dual objective is Y"
    --> Gradient still flows via straight-through
    --> This is exactly what the network needs to make correct
        discrete decisions at inference time
""")
# ============================================================
# Compare Gumbel hard vs soft vs straight-through (no noise)
# ============================================================
print("\n" + "=" * 60)
print("COMPARING GUMBEL HARD vs SOFT vs STRAIGHT-THROUGH")
print("=" * 60)

logits_cmp = torch.tensor([0.5, 2.1, 1.9, 0.2, -1.0], dtype=torch.float64, requires_grad=True)
N_runs = 10

print(f"\nLogits: {[round(l, 1) for l in logits_cmp.tolist()]}")
print(f"True argmax class: {logits_cmp.argmax().item()} ({class_labels[logits_cmp.argmax().item()]})")
print(f"\nRunning {N_runs} forward passes with same logits:\n")

print(f"{'Run':<5} {'Gumbel-soft':<20} {'Gumbel-hard':<20} {'Straight-through':<20}")
print("-" * 65)

for i in range(N_runs):
    # --- Gumbel soft ---
    y_gs = F.gumbel_softmax(logits_cmp.detach(), tau=1.0, hard=False, dim=-1)
    val_gs = (y_gs * classes).sum().item()

    # --- Gumbel hard ---
    y_gh = F.gumbel_softmax(logits_cmp.detach(), tau=1.0, hard=True, dim=-1)
    val_gh = (y_gh * classes).sum().item()

    # --- Straight-through (no noise, deterministic) ---
    probs_st = torch.softmax(logits_cmp.detach(), dim=-1)
    hard_idx = probs_st.argmax(dim=-1, keepdim=True)
    y_hard_st = torch.zeros_like(probs_st).scatter_(-1, hard_idx, 1.0)
    val_st = (y_hard_st * classes).sum().item()

    # Mark if gumbel hard picked a different class than argmax
    gumbel_class = y_gh.argmax().item()
    argmax_class = logits_cmp.argmax().item()
    noise_flag = " <-- noise changed class!" if gumbel_class != argmax_class else ""

    print(f"{i:<5} {val_gs:<20.6f} {val_gh:<20.6f} {val_st:<20.6f}{noise_flag}")

print(f"\nObservations:")
print(f"  Gumbel soft:        always a blended value — never a real class")
print(f"  Gumbel hard:        real class BUT randomly switches due to noise")
print(f"  Straight-through:   always the same real class (argmax) — deterministic")

# ============================================================
# Show gradient difference — which gives cleaner signal?
# ============================================================
print("\n" + "=" * 60)
print("GRADIENT COMPARISON — 5 passes, watch gradient consistency")
print("=" * 60)

for method in ["gumbel_soft", "gumbel_hard", "straight_through"]:
    grads = []
    vals  = []
    for _ in range(5):
        logits_g = torch.tensor([0.5, 2.1, 1.9, 0.2, -1.0],
                                 dtype=torch.float64, requires_grad=True)
        if method == "gumbel_soft":
            y = F.gumbel_softmax(logits_g, tau=1.0, hard=False, dim=-1)
        elif method == "gumbel_hard":
            y = F.gumbel_softmax(logits_g, tau=1.0, hard=True, dim=-1)
        else:
            probs = torch.softmax(logits_g, dim=-1)
            hard_idx = probs.argmax(dim=-1, keepdim=True)
            y_hard = torch.zeros_like(probs).scatter_(-1, hard_idx, 1.0)
            y = y_hard - probs.detach() + probs

        val = (y * classes).sum()
        val.backward()
        grads.append(logits_g.grad.tolist())
        vals.append(val.item())

    grad_tensor = torch.tensor(grads)
    grad_std = grad_tensor.std(dim=0)
    print(f"\n  {method}:")
    print(f"    Output values:    {[f'{v:.5f}' for v in vals]}")
    print(f"    Gradient std:     {[f'{s:.4f}' for s in grad_std.tolist()]}")
    print(f"    --> {'HIGH variance — noisy signal' if grad_std.mean() > 0.01 else 'LOW variance — consistent signal'}")

print(f"""
Conclusion:
  Gumbel soft:       blended outputs, moderate gradient variance
  Gumbel hard:       real outputs BUT high gradient variance due to noise
                     → loss jumps because different class selected each pass
                     → this is why dual opt gap was 3000%
  Straight-through:  real outputs, LOW gradient variance
                     → same class selected every pass → consistent gradient
                     → this is what you want for dual training
""")