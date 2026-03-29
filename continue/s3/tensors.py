import torch

# Create a 1D tensor
a = torch.tensor([1.0, 2.0, 3.0])

# Create a 2D tensor (a matrix)
b = torch.tensor([[1.0, 2.0],
                   [3.0, 4.0]])

# Print both, and their shapes
print(a)
print(a.shape)

print(b)
print(b.shape)

# Check where the tensor lives
print(a.device)

# Move tensor to GPU
a_gpu = a.to("cuda")
print(a_gpu.device)

# The original is still on CPU
print(a.device)

"""-----------------------------------------------"""

# A tensor PyTorch will watch
x = torch.tensor([3.0], requires_grad=True)

# Do some math to it
y = x * 2
z = y + 1

# Ask PyTorch to calculate gradients
# This walks backwards through the graph
z.backward()

# The gradient of z with respect to x
print(x.grad)