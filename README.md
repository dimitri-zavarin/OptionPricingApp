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
* $\vec{\rho}$: $N \times 1$ vector of asset-specific leverage correlations $\in [-1, 1]$.
* $\Sigma$: $N \times N$ empirical correlation matrix of equity log-returns.
* $L$: $N \times N$ lower triangular Cholesky decomposition matrix such that $\Sigma = LL^T$.

---

### 2. Calibration Pipeline

#### 2.1 Two-Stage Hybrid Approach: Structural Constraints + Statistical Weighting

The `MarketDataFetcher/data_fetcher.py` pipeline combines **domain knowledge** (user-provided network structure) with **statistical validation** (constrained VAR-based spillover coefficients):

**Stage 1: Equity Return Correlations (Cholesky Decomposition)**

1. Fetch historical equity prices from Yahoo Finance for the $N$ assets (typically 252 trading days)
2. Compute log-returns: $r_{i,t} = \ln(P_{i,t} / P_{i,t-1})$
3. Calculate correlation matrix: $\Sigma = \text{Corr}(\{r_{i,t}\})$
4. Ensure positive semi-definiteness via eigenvalue correction
5. Compute Cholesky decomposition: $L = \text{cholesky}(\Sigma)$
6. Export to `cholesky_L_matrix.csv`

**Stage 2: Network Structure via Constrained VAR(1)**

The framework uses a **three-step process** to construct the directed network matrix $W$:

**Step 2a: User-Defined Structural Mask** (Binary, Domain-Knowledge)

User creates `skeleton_W_matrix.csv` as an edge list or adjacency matrix specifying which economic relationships are meaningful:

```
From,To
A,B
A,C
B,C
```

This prevents spurious empirical correlations that lack economic justification.

**Step 2b: Constrained VAR(1) Coefficient Extraction**

1. Fetch historical implied volatility (IV) data from DoltHub for the same $N$ assets
2. Fit a **Constrained VAR(1) model** to IV changes where:
   - Only relationships allowed by the skeleton mask receive non-zero coefficients
   - Disallowed edges are exactly 0 during OLS estimation
3. Extract the VAR(1) coefficient matrix $A_1$
4. Take absolute values: $W_{\text{raw}} = |A_1|$ (spillovers are magnitudes)
5. Apply small threshold to remove numerical noise: $W_{\text{raw}}[W_{\text{raw}} < 0.001] = 0$
6. Normalize rows: $W[i,:] = W_{\text{raw}}[i,:] / \sum_j W_{\text{raw}}[i,j]$ (row-stochastic)
7. Export to `calibrated_W_matrix.csv`

**Interpretation**: The VAR(1) coefficient $A_1[i,j]$ represents the average effect of a 1-unit shock in asset $j$'s IV on asset $i$'s IV one step ahead, **subject to the structural constraint imposed by the skeleton**. This ensures:

- ✅ Structural integrity: Only skeleton-approved relationships can have non-zero effects
- ✅ Statistical grounding: Weights reflect actual historical IV spillover dynamics  
- ✅ Simplicity: VAR is the standard econometric tool for multivariate time-series

#### 2.2 Data Separation Rationale

The framework uses **two distinct data sources** to model complementary phenomena:

| Matrix | Data Source | Mechanism | Purpose |
|--------|-------------|-----------|---------|
| **L** (Cholesky) | Equity prices (log-returns) | Return correlations | Fundamental business cycle synchronization; immediate price co-movements |
| **W** (Network) | Implied volatility (IV) | Spillover contagion | Volatility clustering, regimes, and cross-asset propagation |

---

### 3. Discretization & Path Simulation Algorithm

Given initial conditions $\vec{S}_0$ and $\vec{X}_0$, for each time step $t \in \{1, 2, \dots, T\}$:

**Step 1: Generate Independent Standard Normal Shocks**

$$\vec{Z}_1, \vec{Z}_2 \sim \mathcal{N}(\vec{0}, I_N)$$

**Step 2: Apply Cholesky Transformation**

$$\vec{Z}_1^{\text{corr}} = L \, \vec{Z}_1$$

This vector is used for **both** asset return and volatility processes.

**Step 3: Construct Brownian Increments**

$$\Delta \vec{W}_t^S = \sqrt{\Delta t} \, \vec{Z}_1^{\text{corr}}$$

$$\Delta \vec{W}_t^v = \vec{\rho} \odot \sqrt{\Delta t} \, \vec{Z}_1^{\text{corr}} + \sqrt{1 - \vec{\rho}^2} \odot \sqrt{\Delta t} \, \vec{Z}_2$$

**Step 4: Compute Spatiotemporal Volatility Contagion**

$$\vec{N}_t = W\vec{X}_{t-1}$$

$$\vec{\Theta}_t = \vec{\theta} + \vec{\gamma} \odot (\vec{N}_t - \vec{X}_{t-1})$$

**Step 5: Update Network Log-Variance (Euler-Maruyama)**

$$\vec{X}_t = \vec{X}_{t-1} + \vec{\kappa} \odot \left( \vec{\Theta}_t - \vec{X}_{t-1} \right) \Delta t + \vec{\xi} \odot \Delta \vec{W}_t^v$$

**Step 6: Update Asset Prices (Exact Solution)**

$$\vec{S}_t = \vec{S}_{t-1} \odot \exp\left[ \left( \vec{r} - \vec{q} - \frac{1}{2}\exp(\vec{X}_{t-1}) \right) \Delta t + \sqrt{\exp(\vec{X}_{t-1})} \odot \Delta \vec{W}_t^S \right]$$

---

### 4. Option Pricing: Longstaff-Schwartz American Pricer

For American options, we employ the **Longstaff-Schwartz (LS) regression method** with **adaptive basis functions**:

**Basis Function Selection**:

- **4-term basis** (isolated assets): constant, log-moneyness, convexity, local variance
- **5-term basis** (networked assets): adds network variance contagion term $N_{x,\text{exo}}(t) = \sum_{j \neq i} W[i,j] \cdot X_j(t)$

The basis adapts dynamically based on whether an asset receives spillover inputs from neighbors.

---

### 5. C++ Simulation Engine

The `OptionPricingApp/` directory contains the production engine:

- `network_simulator.h/cpp`: Monte Carlo path generator with Cholesky correlations and network spillovers
- `network_pricers.h`: Template-based `EuropeanPricer` and `AmericanPricer` classes
- `simulation_runner.h`: Unified wrapper for batch simulation and pricing
- `option.h`: Payoff definitions (Call/Put × European/American)
- `simulation_config.h`: Centralized parameter management

---

### 6. Workflow Example

**Step 1: Generate calibration matrices**

cd MarketDataFetcher python data_fetcher.py

**Outputs: cholesky_L_matrix.csv, calibrated_W_matrix.csv, W_matrix_network.png**

**Step 2: Build and run the pricer**

cd ../OptionPricingApp cmake -B build && cmake --build build --config Release ./build/OptionPricingApp.exe

---

### 7. References

**Model Inspiration**:

- Heston (1993): Stochastic volatility
- Diebold & Yilmaz (2012): Spillover indices (theoretical foundation for $W$ construction)
- Longstaff & Schwartz (2001): Regression-based American option pricing