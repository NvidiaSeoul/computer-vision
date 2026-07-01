"""results/ 시각화 재현 스크립트 — scikit-learn + matplotlib
실행: python make_figures.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

R = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(R, exist_ok=True)
dig = load_digits()

fig, axs = plt.subplots(4, 10, figsize=(10, 4.2))
for i, ax in enumerate(axs.ravel()):
    ax.imshow(dig.images[i], cmap="gray"); ax.axis("off"); ax.set_title(str(dig.target[i]), fontsize=8)
fig.suptitle("Handwritten Digits (sklearn load_digits, 8x8)")
fig.tight_layout(); fig.savefig(f"{R}/digits_samples.png", dpi=120); plt.close(fig)

p = PCA(2).fit_transform(dig.data)
fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.scatter(p[:, 0], p[:, 1], c=dig.target, cmap="tab10", s=12, alpha=.7)
ax.set(title="Digits — PCA 2D projection", xlabel="PC1", ylabel="PC2")
plt.colorbar(sc, ax=ax, label="digit")
fig.tight_layout(); fig.savefig(f"{R}/digits_pca.png", dpi=120); plt.close(fig)
print("saved figures to", R)
