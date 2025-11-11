#!/usr/bin/env python3
"""Test script to verify action coefficient derivation."""

import numpy as np

from datasets.config import BLUE, RED, WHITE
from datasets.transforms import compute_action_coefficients


def test_basis_vectors():
    """Test that basis vectors return expected coefficients."""
    # Test RED vector
    red_coeffs = compute_action_coefficients(RED)
    assert np.allclose(red_coeffs, [1, 0, 0], atol=1e-10)

    # Test WHITE vector
    white_coeffs = compute_action_coefficients(WHITE)
    assert np.allclose(white_coeffs, [0, 1, 0], atol=1e-10)

    # Test BLUE vector
    blue_coeffs = compute_action_coefficients(BLUE)
    assert np.allclose(blue_coeffs, [0, 0, 1], atol=1e-10)


def test_linear_combination():
    """Test that linear combinations work correctly."""
    # Test: 0.5 * RED + 0.5 * WHITE
    test_action = 0.5 * RED + 0.5 * WHITE
    coeffs = compute_action_coefficients(test_action)
    assert np.allclose(coeffs, [0.5, 0.5, 0], atol=1e-10)

    # Test: 0.3 * RED + 0.3 * WHITE + 0.4 * BLUE
    test_action2 = 0.3 * RED + 0.3 * WHITE + 0.4 * BLUE
    coeffs2 = compute_action_coefficients(test_action2)
    assert np.allclose(coeffs2, [0.3, 0.3, 0.4], atol=1e-10)


def test_reconstruction():
    """Test that we can reconstruct actions from coefficients."""
    test_actions = [
        RED,
        WHITE,
        BLUE,
        0.5 * RED + 0.5 * WHITE,
        0.25 * RED + 0.25 * WHITE + 0.5 * BLUE,
    ]

    basis = np.column_stack([RED, WHITE, BLUE])

    for action in test_actions:
        coeffs = compute_action_coefficients(action)
        reconstructed = basis @ coeffs
        assert np.allclose(action, reconstructed, atol=1e-10)
