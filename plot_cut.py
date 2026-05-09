import torch
import torch.nn.functional as F

torch.set_printoptions(precision=4, sci_mode=False)

# Suppose these are your 5 possible lambda classes
classes = torch.tensor([-0.0001, -0.005, -0.01, -0.05, -10.0])

# Example logits for one lambda variable
examples = {
    "uniform_uncertain": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]),
    "slightly_prefers_class_1": torch.tensor([0.0, 1.0, 0.5, 0.0, -1.0]),
    "two_similar_classes": torch.tensor([0.0, 3.0, 3.0, 0.0, -1.0]),
    "very_confident_class_1": torch.tensor([0.0, 5.0, 0.5, -1.0, -3.0]),
    "very_confident_unmet_demand": torch.tensor([-3.0, -2.0, -1.0, 0.0, 6.0]),
}

def entropy_from_logits(logits):
    probs = F.softmax(logits, dim=-1)
    out = torch.sum(probs * classes)
    entropy = -(probs * torch.log(probs + 1e-12)).sum()
    expected_lambda = (probs * classes).sum()
    hard_lambda = classes[probs.argmax()]
    confidence = probs.max()
    return probs, entropy, expected_lambda, hard_lambda, confidence, out

for name, logits in examples.items():
    probs, entropy, expected_lambda, hard_lambda, confidence, out = entropy_from_logits(logits)

    print(f"\n{name}")
    print(f"logits:          {logits}")
    print(f"probs:           {probs}")
    print(f"out:             {out.item():.4f}")
    print(f"entropy:         {entropy.item():.4f}")
    print(f"max confidence:  {confidence.item():.4f}")
    print(f"E[lambda]:       {expected_lambda.item():.6f}")
    print(f"hard lambda:     {hard_lambda.item():.6f}")


    ''''
    
    '''