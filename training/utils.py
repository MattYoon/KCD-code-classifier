import torch

def top_n_accuracy(preds, target, n=3):
    batch_size = preds.shape[0]
    _, top_n_preds = torch.topk(preds, k=n, dim=1)
    correct = 0
    for i in range(batch_size):
        if target[i] in top_n_preds[i]:
            correct += 1
    return torch.tensor(correct / batch_size)
