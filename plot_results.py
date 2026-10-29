import matplotlib.pyplot as plt
import numpy as np

# Empirical data from unique runs
ip17_m = [37, 18, 102, 21, 17, 14, 19, 20, 13, 23, 24, 16, 18]
ip15_m = [19, 26, 20, 18, 24, 14, 17, 14, 18, 15, 20, 16, 18, 16, 13, 34, 10]

K = 10
ip17_eps = [(m - K) / K for m in ip17_m]
ip15_eps = [(m - K) / K for m in ip15_m]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Plot 1: Boxplot of Overhead Epsilon
data_to_plot = [ip15_eps, ip17_eps]
bp = axes[0].boxplot(
    data_to_plot,
    tick_labels=["iPhone 15\n(N=17)", "iPhone 17\n(N=13)"],
    patch_artist=True,
    showmeans=True,
    meanline=True,
    meanprops={"color": "red", "linewidth": 2, "linestyle": "--"},
    medianprops={"color": "black", "linewidth": 2},
)

colors = ["#4C72B0", "#55A868"]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.65)

axes[0].set_ylabel(r"Overhead Factor $\epsilon = (M - K) / K$", fontsize=11)
axes[0].set_title("Decoding Overhead Factor Across Devices", fontsize=12, fontweight="bold")
axes[0].axhline(0.0, color="gray", linestyle=":", linewidth=1.2, label="Ideal Lower Bound")
axes[0].grid(True, linestyle="--", alpha=0.5)
axes[0].legend(loc="upper right")

# Plot 2: Empirical Cumulative Distribution Function (ECDF)
x_15 = np.sort(ip15_eps)
y_15 = np.arange(1, len(x_15) + 1) / len(x_15)
x_17 = np.sort(ip17_eps)
y_17 = np.arange(1, len(x_17) + 1) / len(x_17)

axes[1].step(x_15, y_15, label="iPhone 15 (Baseline)", where="post", color="#4C72B0", linewidth=2)
axes[1].step(x_17, y_17, label="iPhone 17 (Next-Gen)", where="post", color="#55A868", linewidth=2)
axes[1].set_xlabel(r"Overhead Factor ($\epsilon$)", fontsize=11)
# Fixed \le to \leq inside a raw string
axes[1].set_ylabel(r"Empirical Cumulative Probability $P(E \leq \epsilon)$", fontsize=11)
axes[1].set_title("Convergence Cumulative Distribution Function", fontsize=12, fontweight="bold")
axes[1].set_xlim(-0.05, 3.0)
axes[1].grid(True, linestyle="--", alpha=0.5)
axes[1].legend(loc="lower right")

plt.tight_layout()
plt.savefig("evaluation_benchmark.png", dpi=300)
print("Plots saved successfully as 'evaluation_benchmark.png'")
plt.show()