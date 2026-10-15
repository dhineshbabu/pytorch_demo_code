
import platform
import torch

print("=" * 45)
print("PYTORCH ENVIRONMENT CHECK")
print("=" * 45)

print("Python architecture:", platform.machine())
print("PyTorch version:", torch.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())
print("CUDA available:", torch.cuda.is_available())

# Automatically select the best available device
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("Selected device:", device)

# Create two matrices directly on the GPU
a = torch.randn(1000, 1000, device=device)
b = torch.randn(1000, 1000, device=device)

# Matrix multiplication
c = torch.matmul(a, b)

# Ensure GPU computation has completed
if device.type == "mps":
    torch.mps.synchronize()

print("\nMatrix multiplication completed!")
print("Result shape:", c.shape)
print("Result device:", c.device)
print("All checks passed!")

