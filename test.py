import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import feedforward


inputs = np.arange(0.0, 1.01, 0.05)

lossf = nn.BCELoss()

for input in inputs:
    print(input)
    output = torch.tensor([input], dtype=torch.float32)
    target1 = torch.tensor([0.0], dtype=torch.float32)
    target2 = torch.tensor([1.0], dtype=torch.float32)

    print("1:", lossf(output, target1))
    print("1:", lossf(output, target2))

    output = output.detach().numpy()
    target1 = target1.detach().numpy()
    target2 = target2.detach().numpy()

    print("2:", feedforward.BCELoss(output, target1)[0])
    print("2:", feedforward.BCELoss(output, target2)[0])

print("3:", lossf(torch.tensor([1.0]), torch.tensor([0.0])))
print("4:", feedforward.BCELoss(1.0, 0.0))
