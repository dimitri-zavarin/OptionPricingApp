# OptionPricingApp

## Multi-Asset Spatiotemporal Stochastic Volatility Options Engine with Direct Equity Return Correlations

This engine simulates a multi-asset basket under a network-coupled stochastic volatility framework with empirically-calibrated direct equity return correlations. The model tracks a collection of $N$ assets over time $t \in [0, T]$, where the spot prices are defined by the state vector $\vec{S}_t = [S_{1,t}, S_{2,t}, \dots, S_{N,t}]^T$.

### 1. Continuous-Time Model Dynamics (SDEs)

The joint dynamics of the asset prices and their underlying network-coupled log-variance processes are governed by the following system of stochastic differential equations:

$$d\vec{S}_t = (\vec{r} - \vec{q}) \odot \vec{S}_t \, dt + \sqrt{\exp(\vec{X}_t)} \odot \vec{S}_t \odot d\vec{W}_t^S$$

$$d\vec{X}_t = \vec{\kappa} \odot \left( \vec{\theta} + \vec{\gamma} \odot (W\vec{X}_t - \vec{X}_t) - \vec{X}_t \right) dt + \vec{\xi} \odot d\vec{W}_t^v$$

*(Note: $\odot$ denotes the element-wise Hadamard product).*

The correlated Wiener processes governing the asset returns and volatility shocks are defined in terms of independent standard normal variates $\vec{Z}_1, \vec{Z}_2 \sim \mathcal{N}(\vec{0}, I_N)$ transformed via Cholesky decomposition:

$$d\vec{Z}_1^{\text{corr}} = L \, d\vec{Z}_1$$

where $L$ is the lower triangular Cholesky factor of the equity return correlation matrix $\Sigma$. The transformed shocks are used for both asset returns and volatility:

$$d\vec{W}_t^S = L \, d\vec{Z}_1$$

$$d\vec{W}_t^v = \vec{\rho} \odot L \, d\vec{Z}_1 + \sqrt{1 - \vec{\rho}^2} \odot d\vec{Z}_2$$

**Parameter Definitions:**

* $\vec{r}$: $N \times 1$ risk-free rate vector.
* $\vec{q}$: $N \times 1$ continuously compounded dividend yield vector.
* $\vec{\theta}$: $N \times 1$ idiosyncratic baseline log-variance target vector.
* $\vec{\kappa}$: $N \times 1$ mean-reversion speed vector.
* $\vec{\gamma}$: $N \times 1$ network volatility sensitivity vector.
* $W$: $N \times N$ directed network edge weight matrix (rows normalized to sum to 1).
* $\vec{\xi}$: $N \times 1$ volatility-of-volatility vector.
* $\vec{\rho}$: $N \times 1$ vector of asset-specific leverage correlations $\in [-1, 1]$ (each element represents return-variance correlation for that asset).
* $\Sigma$: $N \times N$ empirical correlation matrix of equity log-returns.
* $L$: $N \times N$ lower triangular Cholesky decomposition matrix such that $\Sigma = LL^T$.

---

### 2. Calibration Pipeline

#### 2.1 Hybrid Approach: Structural Mask + Statistical Weighting

The `MarketDataFetcher/data_fetcher.py` pipeline combines **domain knowledge** (user-provided network structure) with **statistical validation** (Diebold-Yilmaz FEVD):

**Stage 1: Equity Return Correlations (Cholesky Decomposition)**

1. **Fetch historical equity prices** from Yahoo Finance for the $N$ assets over a lookback window (typically 252 trading days)
2. **Compute log-returns**: $r_{i,t} = \ln(P_{i,t} / P_{i,t-1})$ for each asset $i$
3. **Calculate correlation matrix**: $\Sigma = \text{Corr}(\{r_{i,t}\})$ (Pearson correlation of returns)
4. **Ensure positive semi-definiteness** via eigenvalue correction if numerical instabilities exist
5. **Compute Cholesky decomposition**: $L = \text{cholesky}(\Sigma)$ such that $\Sigma = LL^T$
6. **Export to CSV**: `cholesky_L_matrix.csv` (lower triangular $N \times N$ matrix)

**Stage 2: Network Structure via Mask + Diebold-Yilmaz Weighting**

The framework uses a **two-step hybrid approach** to construct the directed network matrix $W$:

**Step 2a: User-Defined Structural Mask** (Binary, Non-stochastic)

User creates `skeleton_W_matrix.csv` specifying which relationships are economically meaningful:

```
From,To
A,B
A,C
B,C
```

**Step 2b: Statistical Weighting (Diebold-Yilmaz)**

1. **Fetch historical implied volatility (IV) data** from DoltHub for the same $N$ assets
2. **Fit a VAR(1) model** to the IV time series
3. **Calculate Generalized Forecast Error Variance Decomposition (GFEVD)** using the Diebold-Yilmaz methodology
4. **Merge with Structural Mask**: Combine user-defined binary relationships with Diebold-Yilmaz weights
5. **Enforce sparsity**: retain only top $K$ neighbors per asset (typically $K=3$) to optimize computational efficiency
6. **Ensure row-stochasticity**: normalize each row to sum to 1.0
7. **Export to CSV**: `calibrated_W_matrix.csv` (row-normalized $N \times N$ matrix)

#### 2.2 Data Separation Rationale

The framework uses **two distinct data sources** to model complementary phenomena:

| Matrix | Data Source | Mechanism | Purpose |
|--------|-------------|-----------|---------|
| **L** (Cholesky) | Equity prices (log-returns) | Direct contemporaneous correlation | Captures fundamental business cycle synchronization; affects immediate price co-movements |
| **W** (Network) | Implied volatility (IV) | Spillover amplification | Captures volatility clustering and regimes; affects volatility contagion across assets |

---

### 3. Discretization & Path Simulation Algorithm

To generate discrete Monte Carlo paths, the continuous-time system is approximated over $T$ time steps using an Euler-Maruyama scheme for the log-variance process and an exact exponential time-stepping solution for the asset prices.

Given initial conditions $\vec{S}_0$ and $\vec{X}_0$, for each time step $t \in \{1, 2, \dots, T\}$:

**Step 1: Generate Independent Standard Normal Shocks**

Generate two independent standard normal vectors:
$$\vec{Z}_1, \vec{Z}_2 \sim \mathcal{N}(\vec{0}, I_N)$$

**Step 2: Apply Cholesky Transformation**

Transform the independent shocks via the Cholesky matrix to induce multi-asset return correlations:
$$\vec{Z}_1^{\text{corr}} = L \, \vec{Z}_1$$

This vector is used for **both** asset return and volatility processes.

**Step 3: Construct Brownian Increments**

Construct Brownian increments for asset returns:
$$\Delta \vec{W}_t^S = \sqrt{\Delta t} \, \vec{Z}_1^{\text{corr}}$$

Construct Brownian increments for volatility using the **same transformed shocks** to preserve the Heston leverage effect:
$$\Delta \vec{W}_t^v = \vec{\rho} \odot \sqrt{\Delta t} \, \vec{Z}_1^{\text{corr}} + \sqrt{1 - \vec{\rho}^2} \odot \sqrt{\Delta t} \, \vec{Z}_2$$

**Key Design Principle**: Both processes use $\vec{Z}_1^{\text{corr}}$, ensuring that:

- **Multi-asset return correlations**: Encoded in the $L$ matrix
- **Heston leverage effect**: Preserved via correlation $\rho$ within each asset
- **Cross-asset volatility spillovers**: Modeled through the network matrix $W$ in the variance dynamics

**Step 4: Compute Spatiotemporal Volatility Contagion**

Evaluate the network-weighted volatility spillover:
$$\vec{N}_t = W\vec{X}_{t-1}$$

Compute the dynamic mean-reversion target incorporating both idiosyncratic and network-driven components:
$$\vec{\Theta}_t = \vec{\theta} + \vec{\gamma} \odot (\vec{N}_t - \vec{X}_{t-1})$$

**Step 5: Update Network Log-Variance (Euler-Maruyama)**

$$\vec{X}_t = \vec{X}_{t-1} + \vec{\kappa} \odot \left( \vec{\Theta}_t - \vec{X}_{t-1} \right) \Delta t + \vec{\xi} \odot \Delta \vec{W}_t^v$$

**Step 6: Transform to Real Variance**

$$\vec{V}_{t-1} = \exp(\vec{X}_{t-1})$$

Exponential transformation guarantees strict positivity: $\vec{V}_{t-1} > 0$.

**Step 7: Update Asset Prices (Exact Log-Normal Discretization)**

Construct the risk-neutral drift with dividend yield adjustment and Itô correction:

$$\vec{\mu}_t = \left(\vec{r} - \vec{q} - \frac{1}{2}\vec{V}_{t-1}\right)\Delta t$$

Construct the diffusion component using correlated Brownian increments:

$$\vec{\sigma}_t = \sqrt{\vec{V}_{t-1}} \odot \Delta \vec{W}_t^S$$

Apply the log-normal update:

$$\vec{S}_t = \vec{S}_{t-1} \odot \exp(\vec{\mu}_t + \vec{\sigma}_t)$$

This exact discretization ensures all simulated prices remain strictly positive and achieves zero structural drift for the asset GBM.

---

### 4. Pricing Methods

#### 4.1 European Options

Simple discounted expectation over all Monte Carlo paths:

$$C_{\text{Euro}}^0 = e^{-r T} \mathbb{E}[C(S_T, K)]$$

#### 4.2 American Options (Longstaff-Schwartz LSMC)

Backward induction using least squares regression on polynomial basis functions that include network effects:

$$V_t(\vec{S}_t, \vec{X}_t) = \max\left\{ \text{Payoff}(S_t), \mathbb{E}[V_{t+1} \mid \vec{S}_t, \vec{X}_t] \right\}$$

The continuation value is projected onto a 4-term or 5-term basis:

- **Constant term**: 1
- **Log-moneyness**: $\ln(S/K)$
- **Convexity**: $[\ln(S/K)]^2 - 1$
- **Local volatility**: $X_t$ (log-variance state)
- **Network spillover** (if active): $\sum_j W_{ij} X_{j,t}$ (weighted sum of neighbor variances)

---

### 5. Workflow & User Inputs

**Required User Inputs:**

1. **Ticker list** (e.g., `["AAPL", "MSFT", "JPM", ...]`)
2. **Skeleton network mask** (`skeleton_W_matrix.csv`)
   - Binary matrix specifying which relationships matter
   - User provides domain knowledge (Bloomberg terminal, economic reasoning)
3. **FEVD horizon** (typically 10-20 steps)
   - Longer horizon = stronger spillover effects

---

### 6. Advantages of Hybrid Approach

- **Domain-aware**: Users encode domain knowledge directly (network structure)
- **Statistically validated**: Diebold-Yilmaz provides empirical spillover strengths
- **Parsimonious**: Avoids spurious spillovers by masking irrelevant pairs
- **Flexible**: Skeleton can be updated based on new Bloomberg terminal data
- **Interpretable**: Each edge represents an economically meaningful relationship
- **Robust**: Doesn't require dense historical data for all possible pairs  

---

### 7. Key Implementation Features

- **Log-Normal Exact Discretization**: Eliminates bias drift for underlying asset paths
- **Eigenvalue Correction**: Ensures correlation matrix positive semi-definiteness before Cholesky decomposition
- **Col-Pivot QR for Regression**: Robust least squares solution even when basis functions are nearly collinear
- **Adaptive Basis Functions**: LSMC automatically includes network spillover term only when asset has active neighbors
- **Row-Stochastic W Matrix**: Ensures economic interpretability of volatility spillovers
- **Unified Shock Space**: Both asset returns and volatility leverage use Cholesky-transformed shocks, preserving the Heston framework