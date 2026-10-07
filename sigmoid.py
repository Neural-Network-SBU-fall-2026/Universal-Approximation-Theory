import numpy as np
import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Generate x values
x = np.linspace(-10, 10, 500)
y = sigmoid(x)

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(x, y, label=r"$\sigma(x) = \frac{1}{1 + e^{-x}}$", color="blue", linewidth=2)

# Reference lines at y=0, y=0.5, y=1, and x=0
plt.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
plt.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
plt.axhline(1, color="gray", linestyle=":", linewidth=1, alpha=0.5)
plt.axvline(0, color="gray", linestyle=":", linewidth=1, alpha=0.5)

# Labels and title
plt.title("Sigmoid Function", fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel(r"$\sigma(x)$", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# Set y-limits slightly beyond [0, 1] for visual clarity
plt.ylim(-0.05, 1.05)

plt.tight_layout()
plt.show()