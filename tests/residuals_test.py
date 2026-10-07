import matplotlib.pyplot as plt
import numpy as np

from flowshoprl import distributions

mu = 0
sigma = 1
scale = 1
k = 2
q = 0.5

ln = distributions.LogNormal(mu=mu, sigma=sigma)
er = distributions.Erlang(scale=scale, k=k)

X = np.linspace(0, 2, 1000)
ln_out = np.zeros(np.size(X))
er_out = np.zeros(np.size(X))

for i, x in enumerate(X):
    print(x)
    ln_out[i] = ln.residual_q(q, x)
    er_out[i] = er.residual_q(q, x)

plt.plot(X, ln_out)
plt.title(f"Lognormal Conditioned Residual: $\mu={mu}$, $\sigma={sigma}, q={q}$")
plt.xlabel("t")
plt.ylabel("$w : Pr(X<t+w | X>t)=q$")
plt.show()
plt.plot(X, er_out)
plt.xlabel("x")
plt.ylabel("$w : Pr(X<t+w | X>t)=q$")
plt.title(f"Erlang Conditioned Residual: $k={k}$, scale$={scale}, q={q}$")
plt.show()
