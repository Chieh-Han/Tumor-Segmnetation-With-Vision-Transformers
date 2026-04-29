import matplotlib.pyplot as plt
import torch

checkpoint = torch.load("checkpoint_latest.pt", map_location="cpu") # maps CUDA weights back to cpu
train_losses = checkpoint["train_losses"]
epochs = range(1, len(train_losses) + 1)

plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()