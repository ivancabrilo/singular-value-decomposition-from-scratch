
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def svd(A, k=100, iterations=25):
    """appx. first k singular triplets using power iteration"""

    R = A.astype(np.float64).copy()
    m, n = R.shape
    k = min(k, m, n)

    U = np.zeros((m, k))
    s = np.zeros(k)
    Vt = np.zeros((k, n))

    rng = np.random.default_rng(42)

    for i in range(k):
        # try random right singular vector at the start
        v = rng.standard_normal(n)
        v /= np.sqrt(np.sum(v**2))

        # pow iteration
        for _ in range(iterations):
            u = R @ v
            norm_u = np.sqrt(np.sum(u**2))
            if norm_u < 1e-12:
                break
            u /= norm_u

            v = R.T @ u
            norm_v = np.sqrt(np.sum(v**2))
            if norm_v < 1e-12:
                break
            v /= norm_v

        # calc singular value and left singular vector
        u = R @ v
        sigma = np.sqrt(np.sum(u**2))

        if sigma < 1e-12:
            return U[:, :i], s[:i], Vt[:i, :]

        u /= sigma

        # save singular triplet
        U[:, i] = u
        s[i] = sigma
        Vt[i, :] = v

        # remove rank one component
        R -= sigma * np.outer(u, v)

    return U, s, Vt



image = Image.open("egypt.jpg").rotate(-90, expand=True).convert("RGB")


image.thumbnail((600, 600), Image.Resampling.LANCZOS)

A = np.array(image)
k = 100

compressed = np.zeros_like(A, dtype=np.float64)

# SVD on each channel separately
for channel in range(3):
    print(f"Processing channel {channel + 1}/3")

    M = A[:, :, channel]

    U, s, Vt = svd(M, k=k)

    compressed[:, :, channel] = (U * s) @ Vt


compressed = np.clip(compressed, 0, 255)
compressed = np.rint(compressed).astype(np.uint8)

Image.fromarray(compressed).save("svd_colored.png")


fig, ax = plt.subplots(1, 2, figsize=(9, 6))

ax[0].imshow(A)
ax[0].set_title("Original Image")
ax[0].axis("off")

ax[1].imshow(compressed)
ax[1].set_title(f"SVD (k={k})")
ax[1].axis("off")

fig.subplots_adjust(
    left=0.01, right=0.99,
    bottom=0.01, top=0.92,
    wspace=0.04
)

plt.savefig("svd_comparison.png", dpi=200, bbox_inches="tight")
plt.show()
