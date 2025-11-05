"""Model definitions and registry for response curve fitting."""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np


def compute_I1(lmbd: np.ndarray) -> np.ndarray:
    """Return the first invariant $I_1 = \lambda^2 + 2/\lambda$."""
    return np.power(lmbd, 2) + 2.0 / lmbd


def strain_factor_yeoh(lmbd: np.ndarray) -> np.ndarray:
    """Return the Yeoh strain factor $(\lambda - \lambda^{-2})$."""
    return lmbd - np.power(lmbd, -2)


def strain_factor_user_defined(lmbd: np.ndarray, k: float) -> np.ndarray:
    """Return a user-defined strain factor used by custom models."""
    x = lmbd - 1
    return x * np.exp(k * (1 - x) ** 2)


def polynomial_term_3rd(lmbd: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Return the third-order polynomial contribution."""
    I1_minus_3 = compute_I1(lmbd) - 3.0
    return a + 2 * b * I1_minus_3 + 3 * c * (I1_minus_3 ** 2)


def polynomial_term_4th(lmbd: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """Return the fourth-order polynomial contribution."""
    I1_minus_3 = compute_I1(lmbd) - 3.0
    return (
        a
        + 2 * b * I1_minus_3
        + 3 * c * (I1_minus_3 ** 2)
        + 4 * d * (I1_minus_3 ** 3)
    )


def model_yeoh_3rd(epsilon: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Third-order Yeoh model."""
    lmbd = epsilon + 1.0
    strain = strain_factor_yeoh(lmbd)
    poly = polynomial_term_3rd(lmbd, a, b, c)
    return 2.0 * strain * poly


def model_yeoh_4th(epsilon: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """Fourth-order Yeoh model."""
    lmbd = epsilon + 1.0
    strain = strain_factor_yeoh(lmbd)
    poly = polynomial_term_4th(lmbd, a, b, c, d)
    return 2.0 * strain * poly


def model_user_defined(epsilon: np.ndarray, a: float, b: float, c: float, k: float) -> np.ndarray:
    """Custom model with exponential strain factor."""
    lmbd = epsilon + 1.0
    strain_factor = strain_factor_user_defined(lmbd, k)
    return 2 * strain_factor * polynomial_term_3rd(lmbd, a, b, c)


def model_user_defined2(epsilon: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Alternative custom model."""
    lmbd = epsilon + 1.0
    I1 = np.power(lmbd, 2) + np.power(lmbd, -1)
    strain_factor = lmbd - np.power(lmbd, -3)
    polynomial_term = a + 2 * b * (I1 - 3) + 3 * c * (I1 - 3) ** 2
    return 2 * strain_factor * polynomial_term


def model_swift(epsilon: np.ndarray, k: float, eps_0: float, n: float) -> np.ndarray:
    """Swift hardening model."""
    return k * np.power(epsilon + eps_0, n)


def model_power(epsilon: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Power law model."""
    return a * np.power(epsilon, b) + c * epsilon


def model_power2(epsilon: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Power law model with quadratic tail."""
    return a * (1 - np.power(epsilon, -b)) + c * epsilon ** 2


def model_bilinear(epsilon: np.ndarray, E: float, sigma0: float, H: float) -> np.ndarray:
    """Bilinear elastic-plastic model."""
    eps_y = sigma0 / E
    sigma = np.zeros_like(epsilon)

    elastic = epsilon <= eps_y
    plastic = epsilon > eps_y

    sigma[elastic] = E * epsilon[elastic]

    eps_p = epsilon[plastic] - eps_y
    sigma[plastic] = sigma0 + H * eps_p

    return sigma


def model_voce(epsilon: np.ndarray, E: float, sigma0: float, r0: float, r_inf: float) -> np.ndarray:
    """Voce model with exponential saturation."""
    eps_y = sigma0 / E
    sigma = np.zeros_like(epsilon)

    elastic = epsilon <= eps_y
    plastic = epsilon > eps_y

    sigma[elastic] = E * epsilon[elastic]

    eps_p = epsilon[plastic] - eps_y
    b = (E - r0) / r_inf
    sigma[plastic] = sigma0 + r0 * eps_p + r_inf * (1 - np.exp(-b * eps_p))

    return sigma


def model_reduced_voce(epsilon: np.ndarray, E: float, sigma0: float, r0: float) -> np.ndarray:
    """Reduced Voce model with linear hardening."""
    eps_y = sigma0 / E
    sigma = np.zeros_like(epsilon)

    elastic = epsilon <= eps_y
    plastic = epsilon > eps_y

    sigma[elastic] = E * epsilon[elastic]

    eps_p = epsilon[plastic] - eps_y
    sigma[plastic] = sigma0 + r0 * eps_p

    return sigma


BoundsType = Tuple[List[float], List[float]]


FITTING_MODELS: Dict[str, Dict[str, object]] = {
    "yeoh": {
        "name": "Yeoh (3rd order)",
        "function": model_yeoh_3rd,
        "param_names": ["a", "b", "c"],
        "description": "σ = 2(λ - λ⁻²) × [a + 2b(I₁-3) + 3c(I₁-3)²]",
    },
    "yeoh4": {
        "name": "Yeoh (4th order)",
        "function": model_yeoh_4th,
        "param_names": ["a", "b", "c", "d"],
        "description": "σ = 2(λ - λ⁻²) × [a + 2b(I₁-3) + 3c(I₁-3)² + 4d(I₁-3)³]",
    },
    "user_defined": {
        "name": "User Defined Model",
        "function": model_user_defined,
        "param_names": ["a", "b", "c", "k"],
        "description": "σ = 2(λ - 1)exp(k(1-λ)²) × [a + 2b(I₁-3) + 3c(I₁-3)²]",
    },
    "user_defined2": {
        "name": "User Defined Model 2",
        "function": model_user_defined2,
        "param_names": ["a", "b", "c"],
        "description": "σ = 2(λ - λ⁻³) × [a + 2b(I₁-3) + 3c(I₁-3)²]",
    },
    "swift": {
        "name": "Swift Model",
        "function": model_swift,
        "param_names": ["k", "eps_0", "n"],
        "description": "σ = k(ε + ε₀)ⁿ",
    },
    "power": {
        "name": "Power Law Model",
        "function": model_power,
        "param_names": ["a", "b", "c"],
        "description": "σ = aεᵇ + cε",
    },
    "power2": {
        "name": "Power Law Model 2",
        "function": model_power2,
        "param_names": ["a", "b", "c"],
        "description": "σ = a(1 - ε⁻ᵇ) + cε²",
    },
    "voce": {
        "name": "Voce Model",
        "function": model_voce,
        "param_names": ["E", r"$\sigma_0$", r"$R_0$", r"$R_\infty$"],
        "bounds": ([0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf]),
        "description": "Voce model with elastic-plastic transition",
    },
    "reduced_voce": {
        "name": "Reduced Voce Model",
        "function": model_reduced_voce,
        "param_names": ["E", r"$\sigma_0$", r"$R_0$"],
        "description": "Reduced Voce model with linear hardening",
    },
    "bilinear": {
        "name": "Bilinear Model",
        "function": model_bilinear,
        "param_names": ["E", r"$\sigma_0$", "H"],
        "bounds": ([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf]),
        "description": "Bilinear elastic-plastic model with linear hardening",
    },
}


def _sanitize_param_name(name: str) -> str:
    """Return a sanitized version of a parameter name without LaTeX markup."""
    cleaned = name.replace("$", "").replace("\\", "")
    return cleaned.replace("{", "").replace("}", "")


for _model_name, _model_info in FITTING_MODELS.items():
    raw_names = _model_info.get("param_names", [])
    safe_names = [_sanitize_param_name(raw) for raw in raw_names]
    _model_info["param_safe_names"] = safe_names
    
    # Calculate and set number of parameters
    n_params = len(raw_names)
    _model_info["n_params"] = n_params

    # Ensure bounds are set
    if "bounds" not in _model_info:
        _model_info["bounds"] = ([-np.inf] * n_params, [np.inf] * n_params)

__all__ = [
    "FITTING_MODELS",
    "model_bilinear",
    "model_power",
    "model_power2",
    "model_reduced_voce",
    "model_swift",
    "model_user_defined",
    "model_user_defined2",
    "model_voce",
]
