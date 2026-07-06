# OptionPricingApp

## Spatiotemporal Network Heston Model

This engine simulates a multi-asset basket under a network-coupled stochastic volatility framework. The model tracks a collection of $N$ assets over time $t \in [0, T]$, where the spot prices are defined by the state vector $\vec{S}_t = [S_{1,t}, S_{2,t}, \dots, S_{N,t}]^T$.

### 1. Continuous-Time Model Dynamics (SDEs)

The joint dynamics of the asset prices and their underlying network-coupled log-variance processes are governed by the following system of stochastic differential equations:

$$d\vec{S}_t = (\vec{r} - \vec{q}) \odot \vec{S}_t \, dt + \sqrt{\exp(\vec{X}_t)} \odot \vec{S}_t \odot d\vec{W}_t^S$$

$$d\vec{X}_t = \vec{K} \odot \left( \vec{\theta} + \Gamma(W\vec{X}_t - \vec{X}_t) - \vec{X}_t \right) dt + \vec{\xi} \odot d\vec{W}_t^v$$

*(Note: $\odot$ denotes the element-wise Hadamard product).*

The correlated Wiener processes governing the asset returns and volatility shocks are defined as:

$$d\vec{W}_t^v = \rho \, d\vec{W}_t^S + \sqrt{1 - \rho^2} \, d\vec{Z}_t$$

**Parameter Definitions:**
* $\vec{r}$: $N \times 1$ risk-free rate vector.
* $\vec{q}$: $N \times 1$ continuously compounded dividend yield vector.
* $\vec{\theta}$: $N \times 1$ idiosyncratic baseline log-variance target vector.
* $\vec{K}$: $N \times 1$ mean-reversion speed vector.
* $\Gamma$: $N \times N$ diagonal network volatility sensitivity matrix ($\text{diag}(\gamma_1, \dots, \gamma_N)$).
* $W$: $N \times N$ spatial network edge weight matrix (rows normalized to sum to 1).
* $\vec{\xi}$: $N \times 1$ volatility-of-volatility vector.
* $\rho$: Scalar $\in [-1, 1]$ representing the asymmetric return-variance correlation (leverage effect).

---

### 2. Discretization & Path Simulation Algorithm

To generate discrete Monte Carlo paths, the continuous-time system is approximated over $T$ steps using an Euler-Maruyama scheme for the log-variance process and an exact exponential time-stepping solution for the asset prices.

Given initial conditions $\vec{S}_0$ and $\vec{X}_0$, for each time step $t \in \{1, 2, \dots, T\}$:

**Step 1: Draw Independent Shocks**

Generate two independent standard normal vectors:
$$\vec{Z}_1, \vec{Z}_2 \sim \mathcal{N}(\vec{0}, I)$$

**Step 2: Construct Correlated Brownian Increments**
$$\Delta \vec{W}_t^S = \sqrt{\Delta t} \, \vec{Z}_1$$
$$\Delta \vec{W}_t^v = \rho \sqrt{\Delta t} \, \vec{Z}_1 + \sqrt{1 - \rho^2} \sqrt{\Delta t} \, \vec{Z}_2$$

**Step 3: Update Network Log-Variance (Euler-Maruyama)**
$$\vec{X}_t = \vec{X}_{t-1} + \vec{K} \odot \left( \vec{\theta} + \Gamma(W\vec{X}_{t-1} - \vec{X}_{t-1}) - \vec{X}_{t-1} \right) \Delta t + \vec{\xi} \odot \Delta \vec{W}_t^v$$

**Step 4: Transform to Real Variance**
$$\vec{V}_{t-1} = \exp(\vec{X}_{t-1})$$

**Step 5: Update Asset Prices (Exact GBM Discretization)**
$$\vec{S}_t = \vec{S}_{t-1} \odot \exp\left( \left(\vec{r} - \vec{q} - \frac{1}{2}\vec{V}_{t-1}\right)\Delta t + \sqrt{\vec{V}_{t-1}} \odot \Delta \vec{W}_t^S \right)$$