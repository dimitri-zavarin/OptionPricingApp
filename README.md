# OptionPricingApp

## Multi-Asset Spatiotemporal Stochastic Volatility Options Engine with Direct Equity Return Correlations

This engine simulates a multi-asset basket under a network-coupled stochastic volatility framework with empirically-calibrated direct equity return correlations. The model tracks a collection of $N$ assets over time $t \in [0, T]$, where the spot prices are defined by the state vector $\vec{S}_t = [S_{1,t}, S_{2,t}, \dots, S_{N,t}]^T$.

### 1. Continuous-Time Model Dynamics (SDEs)

The joint dynamics of the asset prices and their underlying network-coupled log-variance processes are governed by the following system of stochastic differential equations:

$$d\vec{S}_t = (\vec{r} - \vec{q}) \odot \vec{S}_t \, dt + \sqrt{\exp(\vec{X}_t)} \odot \vec{S}_t \odot d\vec{W}_t^{S,\text{corr}}$$

$$d\vec{X}_t = \vec{K} \odot \left( \vec{\theta} + \vec{\gamma} \odot (W\vec{X}_t - \vec{X}_t) - \vec{X}_t \right) dt + \vec{\xi} \odot d\vec{W}_t^v$$

*(Note: $\odot$ denotes the element-wise Hadamard product).*

The correlated Wiener processes governing the asset returns and volatility shocks are defined as:

$$d\vec{W}_t^{S,\text{corr}} = L \, d\vec{W}_t^{S,\text{indep}}$$

$$d\vec{W}_t^v = \rho \, d\vec{W}_t^{S,\text{indep}} + \sqrt{1 - \rho^2} \, d\vec{Z}_t$$

where $L$ is the lower triangular Cholesky factor of the equity return correlation matrix $\Sigma$.

**Parameter Definitions:**

* $\vec{r}$: $N \times 1$ risk-free rate vector.
* $\vec{q}$: $N \times 1$ continuously compounded dividend yield vector.
* $\vec{\theta}$: $N \times 1$ idiosyncratic baseline log-variance target vector.
* $\vec{K}$: $N \times 1$ mean-reversion speed vector.
* $\vec{\gamma}$: $N \times 1$ network volatility sensitivity vector.
* $W$: $N \times N$ directed network edge weight matrix (rows normalized to sum to 1).
* $\vec{\xi}$: $N \times 1" volatility-of-volatility vector.
* $\rho$: Scalar $\in [-1, 1]$ representing the asymmetric return-variance correlation (leverage effect).
* $\Sigma": $N \times N$ empirical correlation matrix of equity log-returns.
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

**Step 1: Draw Independent Standard Normal Shocks**

Generate two independent standard normal vectors:
$$\vec{Z}_1, \vec{Z}_2 \sim \mathcal{N}(\vec{0}, I_N)$$

**Step 2: Apply Cholesky Transformation for Correlated Asset Returns**

Transform the first shock vector via the lower triangular Cholesky matrix to induce direct equity return correlations:
$$\vec{Z}_1^{\text{corr}} = L \, \vec{Z}_1$$

where $L$ encodes the empirical correlation structure from historical returns.

**Step 3: Construct Brownian Increments**

$$\Delta \vec{W}_t^{S,\text{corr}} = \sqrt{\Delta t} \, \vec{Z}_1^{\text{corr}}$$

$$\Delta \vec{W}_t^v = \rho \sqrt{\Delta t} \, \vec{Z}_1 + \sqrt{1 - \rho^2} \sqrt{\Delta t} \, \vec{Z}_2$$

**CRITICAL**: The volatility increment $\Delta \vec{W}_t^v$ uses the **original $\vec{Z}_1$** (not $\vec{Z}_1^{\text{corr}}$) to maintain asset-specific Heston leverage effects independently of cross-asset correlations.

**Step 4: Compute Spatiotemporal Volatility Contagion**

Evaluate the network-weighted volatility spillover:
$$\vec{N}_t = W\vec{X}_{t-1}$$

Compute the dynamic mean-reversion target incorporating both idiosyncratic and network-driven components:
$$\vec{\Theta}_t = \vec{\theta} + \vec{\gamma} \odot (\vec{N}_t - \vec{X}_{t-1})$$

**Step 5: Update Network Log-Variance (Euler-Maruyama)**

$$\vec{X}_t = \vec{X}_{t-1} + \vec{K} \odot \left( \vec{\Theta}_t - \vec{X}_{t-1} \right) \Delta t + \vec{\xi} \odot \Delta \vec{W}_t^v$$

**Step 6: Transform to Real Variance**

$$\vec{V}_{t-1} = \exp(\vec{X}_{t-1})$$

Exponential transformation guarantees strict positivity: $\vec{V}_{t-1} > 0$.

**Step 7: Update Asset Prices (Exact Log-Normal Discretization)**

Construct the risk-neutral drift with dividend yield adjustment and Itô correction:

$$\vec{\mu}_t = \left(\vec{r} - \vec{q} - \frac{1}{2}\vec{V}_{t-1}\right)\Delta t$$

Construct the diffusion component using **correlated Brownian increments**:

$$\vec{\sigma}_t = \sqrt{\vec{V}_{t-1}} \odot \Delta \vec{W}_t^{S,\text{corr}}$$

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

### 5. Architecture Overview

The pipeline consists of two main stages:

**Stage 1: Python Market Data Calibration**

    MarketDataFetcher/
    ├── get_equity_price_history()
    │   └── Yahoo Finance ──> cholesky_L_matrix.csv
    ├── calculate_return_correlation_matrix()
    │   └── Computes log-return correlation + Cholesky decomposition
    ├── get_dolthub_iv_history()
    │   └── DoltHub API ──> calibrated_W_matrix.csv
    └── generate_sparse_dynamic_W_matrix()
        └── VAR(1) + Diebold-Yilmaz GFEVD

**Stage 2: C++ Monte Carlo Engine**

    OptionPricingApp/
    ├── config_loader.h/cpp
    │   ├── load_cholesky_matrix() ──> Reads cholesky_L_matrix.csv
    │   └── load_network_weight_matrix() ──> Reads calibrated_W_matrix.csv
    ├── simulation_config.h
    │   └── Stores config_.L and config_.W matrices
    ├── network_simulator.h
    │   ├── Applies Cholesky transformation: z1_corr = L * z1
    │   └── Generates correlated paths with independent leverage effects
    ├── network_pricers.h
    │   ├── EuropeanPricer: Simple discounted payoff
    │   └── AmericanPricer: Longstaff-Schwartz LSMC with network basis
    └── main.cpp
        └── Orchestrates entire pipeline

---

### 6. Key Implementation Details

**Cholesky Transformation in Simulation Loop** (network_simulator.h, Step 2):

    // Apply Cholesky for correlated asset returns
    Eigen::VectorXd z1_correlated = config_.L * z1;
    
    // Construct correlated Brownian increments for asset returns
    Eigen::VectorXd dws = sqrt_dt * z1_correlated;
    
    // Construct volatility increments using original z1 (maintains independence)
    Eigen::VectorXd dwv = sqrt_dt * (config_.rho.cwiseProduct(z1) +
        (1.0 - config_.rho.array().square()).sqrt().matrix().cwiseProduct(z2));

**Key Architectural Principle**: The asset return shocks exhibit empirical correlation structure via $L$, while leverage effects (rho) remain calibrated independently per asset, preventing over-parameterization and ensuring economically interpretable results.

**CSV Loading Logic** (config_loader.cpp):

    - Skips header row (ticker labels)
    - Parses each data row, skipping first column (index names)
    - Validates dimensions match num_assets
    - Returns false if file I/O fails or dimensions mismatch

**LSMC Regression Basis** (network_pricers.h):

    - 4-term basis: [1, ln(S/K), (ln(S/K))^2 - 1, X]
    - 5-term basis (if network-linked): above + sum_j W_ij * X_j
    - Uses ColPivHouseholderQR for numerical stability
    - Adaptive basis selection based on network topology

---

### 7. Numerical Features & Safeguards

- **Log-Normal Exact Discretization**: Eliminates bias drift for underlying asset paths
- **Eigenvalue Correction**: Ensures correlation matrix positive semi-definiteness before Cholesky decomposition
- **Col-Pivot QR for Regression**: Robust least squares solution even when basis functions are nearly collinear
- **Adaptive Basis Functions**: LSMC automatically includes network spillover term only when asset has active neighbors (W_ij > 0)
- **Row-Stochastic W Matrix**: Ensures economic interpretability of volatility spillovers
- **Minimum Sample Size Enforcement**: Skips regression if fewer than num_features + 5 ITM paths available
- **Standard Error Diagnostics**: Rolling output of parameter significance, median betas, and condition numbers