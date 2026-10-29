"""Denoising autoencoder in NumPy: market state -> 8-dim "fingerprint".

The network learns to rebuild a month's full state from a corrupted copy
(noise plus randomly hidden indicators). Its bottleneck has to keep what
matters about the state, which makes distances between fingerprints a
better similarity than raw indicator distances.
"""
import numpy as np


class Encoder:
    def __init__(self, n_in, n_hidden=64, n_latent=8, seed=0):
        rng = np.random.default_rng(seed)

        def layer(a, b):
            return rng.normal(0, np.sqrt(2 / (a + b)), (a, b)), np.zeros(b)

        self.params = [
            *layer(n_in, n_hidden),
            *layer(n_hidden, n_latent),
            *layer(n_latent, n_hidden),
            *layer(n_hidden, n_in),
        ]
        self.rng = rng

    def _forward(self, x):
        W1, b1, W2, b2, W3, b3, W4, b4 = self.params
        h1 = np.tanh(x @ W1 + b1)
        z = h1 @ W2 + b2
        h3 = np.tanh(z @ W3 + b3)
        out = h3 @ W4 + b4
        return h1, z, h3, out

    def encode(self, x):
        return self._forward(x)[1]

    def fit(self, x, epochs=300, batch=64, lr=2e-3, noise=0.35, mask=0.15, l2=1e-4):
        m = [np.zeros_like(p) for p in self.params]
        v = [np.zeros_like(p) for p in self.params]
        step = 0
        n = len(x)
        for _ in range(epochs):
            order = self.rng.permutation(n)
            for i in range(0, n, batch):
                xb = x[order[i:i + batch]]
                keep = self.rng.random(xb.shape) > mask
                xin = (xb + self.rng.normal(0, noise, xb.shape)) * keep
                h1, z, h3, out = self._forward(xin)
                W1, b1, W2, b2, W3, b3, W4, b4 = self.params
                d_out = 2 * (out - xb) / xb.size
                gW4, gb4 = h3.T @ d_out, d_out.sum(0)
                d_h3 = (d_out @ W4.T) * (1 - h3 ** 2)
                gW3, gb3 = z.T @ d_h3, d_h3.sum(0)
                d_z = d_h3 @ W3.T
                gW2, gb2 = h1.T @ d_z, d_z.sum(0)
                d_h1 = (d_z @ W2.T) * (1 - h1 ** 2)
                gW1, gb1 = xin.T @ d_h1, d_h1.sum(0)
                grads = [gW1, gb1, gW2, gb2, gW3, gb3, gW4, gb4]
                step += 1
                for j, (p, g) in enumerate(zip(self.params, grads)):
                    if p.ndim == 2:
                        g = g + l2 * p
                    m[j] = 0.9 * m[j] + 0.1 * g
                    v[j] = 0.999 * v[j] + 0.001 * g * g
                    mh = m[j] / (1 - 0.9 ** step)
                    vh = v[j] / (1 - 0.999 ** step)
                    p -= lr * mh / (np.sqrt(vh) + 1e-8)
        return self

    def loss(self, x):
        return float(np.mean((self._forward(x)[3] - x) ** 2))


def fingerprints(z_fit, z_all, seeds=(0, 1, 2)):
    """Train a small ensemble and stack whitened latents (more stable neighbours)."""
    outs = []
    for s in seeds:
        enc = Encoder(z_fit.shape[1], seed=s).fit(z_fit)
        lat_fit = enc.encode(z_fit)
        mu, sd = lat_fit.mean(0), lat_fit.std(0) + 1e-9
        outs.append((enc.encode(z_all) - mu) / sd)
    return np.hstack(outs)
