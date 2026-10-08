import math

import matplotlib.pyplot as plt
import numpy as np

from flowshoprl import distributions

mean = 2
cv = 1 / math.sqrt(2)
q = 0.5

print(f"mean={mean}, cv={cv}, q={q}")

ln = distributions.LogNormal(mean, cv)
er = distributions.Erlang(mean, cv)

X = np.linspace(0, 10, 1000)
ln_out = np.zeros(np.size(X))
er_out = np.zeros(np.size(X))

for i, x in enumerate(X):
    ln_out[i] = ln.residual_q(q, x)
    er_out[i] = er.residual_q(q, x)

plt.plot(X, ln_out)
plt.title(
    rf"Lognormal Conditioned Residual: $\mu={ln.mu:.2f}$, $\sigma={ln.sigma:.2f}, q={q}$"
)
plt.xlabel("t")
plt.ylabel("$w : Pr(X<t+w | X>t)=q$")
plt.show()
plt.plot(X, er_out)
plt.xlabel("t")
plt.ylabel("$w : Pr(X<t+w | X>t)=q$")
plt.title(
    f"Erlang Conditioned Residual: $k={int(er.k)}$, scale$={er.scale:.2f}, q={q}$"
)
plt.show()
