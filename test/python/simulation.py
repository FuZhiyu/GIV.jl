import numpy as np
import pandas as pd
from scipy.stats import t, norm
from scipy.optimize import root_scalar
from dataclasses import dataclass, field
from typing import Union


@dataclass
class SimParam:
    # Core parameters with defaults
    gamma: float = 0.5
    h: float = 0.2
    M: float = 0.5
    T: int = 60
    K: int = 0
    N: int = 20
    nu: float = np.inf
    NC: int = 2
    sigma_u_curv: float = 0.1
    usupplyshare: float = 0.2
    sigma_p: float = 2.0
    sigma_zeta: float = 1.0
    zeta_s: float = 1.0

    # Calculated parameters
    var_u_share: float = field(init=False)
    constS: np.ndarray = field(init=False)
    sigma_u_vec: np.ndarray = field(init=False)
    DIST: Union[norm, t] = field(init=False)

    def __post_init__(self):
        # Set variance share based on Julia logic
        self.var_u_share = 1.0 if (self.NC == 0 and self.K == 0) else 0.2

        # Solve for size distribution
        self.constS = self._solve_S_for_hhi()

        # Calculate idiosyncratic volatility
        self.sigma_u_vec = self._specify_sigma_i()

        # Set distribution with proper scaling
        if np.isinf(self.nu):
            self.DIST = norm(0, 1)
        else:
            scale = np.sqrt((self.nu - 2) / self.nu)
            self.DIST = t(self.nu, scale=scale)

    def _solve_S_for_hhi(self):
        """Solve for firm size distribution given HHI"""

        def h_from_tp(tp):
            k = np.arange(1, self.N + 1) ** (-1 / tp)
            S = k / k.sum()
            h_prime = np.sqrt((S ** 2).sum() - 1 / self.N)
            return self.h - h_prime

        res = root_scalar(h_from_tp, bracket=[0.1, 5.0], x0=1.0)
        k = np.arange(1, self.N + 1) ** (-1 / res.root)
        return k / k.sum()

    def _specify_sigma_i(self):
        """Calculate idiosyncratic volatility schedule with Julia-compatible scaling"""
        S = self.constS
        sigma_i_sq = np.exp(-self.sigma_u_curv * np.log(S))
        b = (S.T @ S) / (S.T @ np.diag(sigma_i_sq) @ S)
        sigma_i_sq *= b

        if self.usupplyshare > 0:
            sigma_D_sq = np.sum((S * np.sqrt(sigma_i_sq)) ** 2)
            sigma_s_sq = self.usupplyshare * sigma_D_sq / (1 - self.usupplyshare)
            return np.concatenate([np.sqrt(sigma_i_sq), [np.sqrt(sigma_s_sq)]])

        return np.sqrt(sigma_i_sq)


@dataclass
class SimData:
    S: np.ndarray  # (N, T)
    u: np.ndarray  # (N, T)
    m: np.ndarray  # (N, NC)
    C: np.ndarray  # (NC, T)
    Lambda: np.ndarray  # (N, K)
    eta: np.ndarray  # (K, T)
    p: np.ndarray  # (1, T)
    zeta: np.ndarray  # (N,)
    q: np.ndarray  # (N, T)

    def to_dataframe(self):
        """Convert to pandas DataFrame with Julia-compatible structure"""
        N, T = self.q.shape
        df = pd.DataFrame({
            'S': self.S.flatten('F'),
            'u': self.u.flatten('F'),
            'q': self.q.flatten('F'),
            'id': np.tile(np.arange(1, N + 1), T),
            'zeta': np.repeat(self.zeta, T),
            't': np.repeat(np.arange(1, T + 1), N),
            # CORRECTED LINE: Repeat each time period's price N times
            'p': np.repeat(self.p.flatten(), N)
        })

        # Add eta factors with proper normalization
        eta_df = pd.DataFrame(
            np.repeat(self.C.T, N, axis=0),
            columns=[f'eta{i + 1}' for i in range(self.C.shape[0])]
        )

        return pd.concat([df, eta_df], axis=1).assign(
            absS=lambda x: np.abs(x.S)
        )


class SimModel:
    """Julia-compatible simulation model"""

    def __init__(self, **kwargs):
        self.param = SimParam(**kwargs)
        self.data = self._generate_data()

    def _generate_data(self) -> SimData:
        """Core data generation matching Julia implementation"""
        p = self.param
        N = p.N

        # Initialize parameters with Julia-style adjustments
        zeta = np.random.normal(0, p.sigma_zeta, N)
        zeta -= np.sum(zeta * np.abs(p.constS))

        # Handle supply/demand parameters
        supply_flag = p.zeta_s > 0 or p.usupplyshare > 0
        if supply_flag:
            zeta_S = 1 / p.M
            zeta_D = zeta_S - p.zeta_s
            assert zeta_D > 0, "Demand elasticity must be positive"
            zeta += zeta_D
            zeta = np.concatenate([zeta, [p.zeta_s]])
            constS = np.concatenate([p.constS, [-1.0]])
            N += 1
        else:
            zeta += 1 / p.M  # Julia-style in-place addition
            constS = p.constS

        absS = np.abs(constS)

        # Generate random components with proper normalization
        u = p.DIST.rvs(size=(N, p.T)) * p.sigma_u_vec[:, None]
        m = np.random.normal(size=(N, p.NC))
        C = np.random.normal(size=(p.NC, p.T))

        # Julia-style normalization of C matrix
        if p.NC > 0:
            try:
                cov_matrix = np.cov(C)
                cholL = np.linalg.cholesky(cov_matrix)
            except np.linalg.LinAlgError:
                # Add small regularization for numerical stability
                cov_matrix += 1e-6 * np.eye(cov_matrix.shape[0])
                cholL = np.linalg.cholesky(cov_matrix)
            C = np.linalg.inv(cholL.T) @ C

        Lambda = np.random.normal(size=(N, p.K))
        eta = np.random.normal(size=(p.K, p.T))

        # Variance scaling matching Julia implementation
        if p.NC > 0 or p.K > 0:
            common_shocks = m @ C + Lambda @ eta
            var_ratio = (np.var(absS.T @ u) / np.var(absS.T @ common_shocks))
            scale = np.sqrt(p.var_u_share / (var_ratio * (1 - p.var_u_share)))
            u *= scale
            p.sigma_u_vec = p.sigma_u_vec * scale

        # Final shock construction with Julia-compatible scaling
        shock = u + m @ C + Lambda @ eta
        net_shock = absS.T @ shock

        # Scale to match target sigma_p
        netshockscale = np.sqrt(np.var(net_shock * p.M) / p.sigma_p ** 2)
        u /= netshockscale
        m /= netshockscale
        Lambda /= netshockscale
        shock = u + m @ C + Lambda @ eta
        net_shock = absS.T @ shock

        p_value = (net_shock * p.M).reshape(1, -1)
        q = shock - zeta[:, None] * p_value

        return SimData(
            S=np.tile(constS, (p.T, 1)).T,
            u=u,
            m=m,
            C=C,
            Lambda=Lambda,
            eta=eta,
            p=p_value,
            zeta=zeta,
            q=q
        )


# Example usage with proper parameters
if __name__ == "__main__":
    model = SimModel(
        T=400,
        N=100,
        usupplyshare=0.0,
        h=0.3,
        sigma_u_curv=0.2,
        zeta_s=0.0,
        NC=2,
        M=0.5,
        sigma_zeta=0.0
    )
    df = model.data.to_dataframe()
    print("Generated DataFrame:")
    print(df.head(10))
    print("\nDimensions:", df.shape)