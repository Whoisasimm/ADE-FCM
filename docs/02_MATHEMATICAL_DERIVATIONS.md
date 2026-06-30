# Mathematical Derivations for ADE-FCM

## 1. Standard Fuzzy C-Means (FCM)

### 1.1 Objective Function
$$J_{FCM} = \sum_{i=1}^{N} \sum_{j=1}^{C} u_{ij}^m \|x_i - v_j\|^2$$

**Constraints:**
- $\sum_{j=1}^{C} u_{ij} = 1, \quad \forall i$
- $0 \leq u_{ij} \leq 1$
- $\sum_{i=1}^{N} u_{ij} > 0, \quad \forall j$

### 1.2 Lagrangian Optimization
$$\mathcal{L} = \sum_{i=1}^{N} \sum_{j=1}^{C} u_{ij}^m d_{ij}^2 + \sum_{i=1}^{N} \lambda_i \left(1 - \sum_{j=1}^{C} u_{ij}\right)$$

Partial derivative w.r.t $u_{ij}$:
$$\frac{\partial \mathcal{L}}{\partial u_{ij}} = m u_{ij}^{m-1} d_{ij}^2 - \lambda_i = 0$$
$$u_{ij} = \left(\frac{\lambda_i}{m d_{ij}^2}\right)^{\frac{1}{m-1}}$$

Using constraint $\sum_k u_{ik} = 1$:
$$u_{ij} = \frac{1}{\sum_{k=1}^{C} \left(\frac{d_{ij}}{d_{ik}}\right)^{\frac{2}{m-1}}}$$

### 1.3 Cluster Center Update
$$\frac{\partial J}{\partial v_j} = -2 \sum_{i=1}^{N} u_{ij}^m (x_i - v_j) = 0$$
$$v_j = \frac{\sum_{i=1}^{N} u_{ij}^m x_i}{\sum_{i=1}^{N} u_{ij}^m}$$

## 2. Fuzzy C-Median (FCLM) - Paper's Algorithm

### 2.1 Median-Based Distance
$$D_i = \text{Median}\{D_{ij}(S_k - S_i) \cdot u_{ij}\}, \quad \forall i \neq k$$

### 2.2 Center Selection via Argmin
$$p = \operatorname{Argmin}\{D_i : n\}, \quad \forall i = 1 \ldots n$$

### 2.3 Membership Update (same as FCM)
$$u_{ij} = \frac{1}{\sum_{k=1}^{C} \left(\frac{d_{ij}}{d_{ik}}\right)^{\frac{2}{m-1}}}$$

### 2.4 Convergence Criterion
$$\|U^{(k+1)} - U^{(k)}\| < \varepsilon$$

## 3. ADE-FCM Novel Contributions

### 3.1 KMeans++ Initialization
Probability of selecting point $x_i$ as centroid:
$$P(x_i) = \frac{D(x_i)^2}{\sum_{j=1}^{N} D(x_j)^2}$$

### 3.2 Density-Based Initialization
$$\rho_i = \sum_{j=1}^{N} \exp\left(-\frac{\|x_i - x_j\|^2}{d_c^2}\right)$$

### 3.3 Adaptive Fuzzifier
$$m^{(t)} = m_{\min} + (m_{\max} - m_{\min}) \cdot \exp\left(-\alpha \cdot \frac{t}{T}\right)$$

### 3.4 Confidence-Weighted Membership
$$\text{conf}_i = 1 - \frac{2}{\pi} \arctan\left(\frac{\sigma_i}{\mu_i}\right)$$

### 3.5 Automatic Cluster Discovery
$$K_{\text{opt}} = \text{argmax}_k \left[ w_1 \cdot S(k) - w_2 \cdot DB(k) + w_3 \cdot \text{Gap}(k) \right]$$

### 3.6 Outlier-Robust Membership
$$O_i = \sum_{j=1}^{C} u_{ij}^m \cdot d(x_i, v_j)$$

### 3.7 Dynamic Convergence Threshold
$$\varepsilon^{(t)} = \varepsilon_0 \cdot \exp\left(-\beta \cdot \frac{t}{T}\right)$$

### 3.8 ADE-FCM Complete Objective Function
$$J_{ADE-FCM} = \sum_{i=1}^{N} \sum_{j=1}^{C} u_{ij}^{m^{(t)}} \cdot d(x_i, v_j) + \lambda \sum_{i=1}^{N} \sum_{j=1}^{C} u_{ij} \log u_{ij}$$
