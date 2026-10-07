import numpy as np
import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

# Define the sigmoid function with a coefficient
def sigmoid_coefficient(x, coefficient):
    return 1 / (1 + np.exp(-coefficient * x))

# Generate x values
x = np.linspace(-10, 10, 500)

# List of coefficients to compare
coefficients = [1, 2, 5, 10]

# Create the plot
plt.figure(figsize=(10, 6))

for k in coefficients:
    y = sigmoid_coefficient(x, k)
    plt.plot(x, y, linewidth=2, label=rf"$k = {k}$")

# Reference lines
plt.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
plt.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
plt.axhline(1, color="gray", linestyle=":", linewidth=1, alpha=0.5)
plt.axvline(0, color="gray", linestyle=":", linewidth=1, alpha=0.5)

# Labels and title
plt.title(r"Sigmoid Function $\sigma(x) = \frac{1}{1 + e^{-kx}}$ for different $k$", fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel(r"$\sigma(x)$", fontsize=12)
plt.legend(fontsize=11, title="Coefficient")
plt.grid(True, alpha=0.3)
plt.ylim(-0.05, 1.05)

plt.tight_layout()
plt.show()