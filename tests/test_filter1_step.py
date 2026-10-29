"""Prueba 4 de Milca: un paso del filtro de Kalman 1 calculado a mano con las fórmulas del profesor."""
import numpy as np

from kalman.filter1_weights import kalman_filter


def test_one_step_by_hand_matches_code():
    y, C = np.array([0.20]), np.array([[0.50, -0.10, 0.30]])
    Q, R = np.diag([1e-4, 2e-4, 3e-4]), 1.6e-3
    z0, P0 = np.array([0.30, -0.20, 0.10]), np.diag([0.5, 0.2, 0.1])
    A, I = np.eye(3), np.eye(3)
    z_prior = A @ z0                                  # ẑ_{t|t-1} = A·ẑ_{t-1|t-1}
    P_prior = A @ P0 @ A.T + Q                        # P_{t|t-1} = A·P·Aᵀ + Q
    innovation = y[0] - C[0] @ z_prior                # ỹ_t = y_t − C_t·ẑ_{t|t-1}
    S = C[0] @ P_prior @ C[0] + R                     # S_t = C_t·P_{t|t-1}·C_tᵀ + R
    K = P_prior @ C[0] / S                            # K_t = P_{t|t-1}·C_tᵀ·S_t⁻¹
    z_post = z_prior + K * innovation                 # ẑ_{t|t} = ẑ_{t|t-1} + K_t·ỹ_t
    P_post = (I - np.outer(K, C[0])) @ P_prior        # P_{t|t} = (I − K_t·C_t)·P_{t|t-1}

    result = kalman_filter(y, C, Q, R, z0, P0)
    np.testing.assert_allclose(result["z_prior"][0], z_prior)
    np.testing.assert_allclose(result["innovation"][0], innovation)
    np.testing.assert_allclose(result["innovation_var"][0], S)
    np.testing.assert_allclose(result["gain"][0], K)
    np.testing.assert_allclose(result["z_post"][0], z_post)
    np.testing.assert_allclose(result["posvar"][0], np.diag(P_post))
