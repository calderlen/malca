"""Tests for baseline computation functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from malca.baseline import (
    per_camera_gp_baseline,
    per_camera_gp_baseline_masked,
    per_camera_trend_baseline,
)


def make_synthetic_lc(
    n_points: int = 200,
    n_cameras: int = 2,
    base_mag: float = 14.0,
    scatter: float = 0.02,
    error: float = 0.015,
    jd_start: float = 2458000.0,
    cadence: float = 3.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic quiescent light curve with all required columns."""
    rng = np.random.default_rng(seed)
    
    n_per_cam = n_points // n_cameras
    rows = []
    
    for cam in range(n_cameras):
        jd = jd_start + np.arange(n_per_cam) * cadence + rng.uniform(0, 1, n_per_cam)
        mag = base_mag + rng.normal(0, scatter, n_per_cam)
        err = np.full(n_per_cam, error) + rng.uniform(0, 0.005, n_per_cam)
        
        for j, m, e in zip(jd, mag, err):
            rows.append({
                "JD": j, 
                "mag": m, 
                "error": e, 
                "camera#": f"b{cam}",
                "saturated": 0,  # Required by clean_lc
            })
    
    return pd.DataFrame(rows).sort_values("JD").reset_index(drop=True)


def inject_dip(
    df: pd.DataFrame,
    t0: float,
    amplitude: float = 0.5,
    sigma: float = 10.0,
) -> pd.DataFrame:
    """Inject a Gaussian dip into the light curve."""
    df = df.copy()
    gaussian = amplitude * np.exp(-0.5 * ((df["JD"] - t0) / sigma) ** 2)
    df["mag"] = df["mag"] + gaussian  # Positive = fainter = dip
    return df


class TestSigmaFloorConsistency:
    """Test that GP and masked GP produce consistent sigma_eff values."""

    def test_gp_baseline_has_sigma_eff(self):
        """Verify per_camera_gp_baseline produces sigma_eff column."""
        df = make_synthetic_lc()
        result = per_camera_gp_baseline(df, add_sigma_eff_col=True)
        
        assert "sigma_eff" in result.columns
        assert result["sigma_eff"].notna().all()
        assert (result["sigma_eff"] > 0).all()

    def test_gp_masked_baseline_has_sigma_eff(self):
        """Verify per_camera_gp_baseline_masked produces sigma_eff column."""
        df = make_synthetic_lc()
        result = per_camera_gp_baseline_masked(df, add_sigma_eff_col=True)
        
        assert "sigma_eff" in result.columns
        assert result["sigma_eff"].notna().all()
        assert (result["sigma_eff"] > 0).all()

    def test_sigma_eff_similar_on_quiescent_lc(self):
        """Both baselines should give similar sigma_eff on clean data."""
        df = make_synthetic_lc(seed=123)
        
        result_gp = per_camera_gp_baseline(df, add_sigma_eff_col=True)
        result_masked = per_camera_gp_baseline_masked(df, add_sigma_eff_col=True)
        
        # Median sigma_eff should be within 50% of each other
        median_gp = result_gp["sigma_eff"].median()
        median_masked = result_masked["sigma_eff"].median()
        
        ratio = median_masked / median_gp
        assert 0.5 < ratio < 2.0, f"sigma_eff ratio {ratio:.2f} too different"

    def test_sigma_eff_includes_sigma_floor(self):
        """sigma_eff should be larger than just yerr due to sigma_floor."""
        df = make_synthetic_lc()
        result = per_camera_gp_baseline(df, add_sigma_eff_col=True)
        
        # sigma_eff should generally be >= error (due to floor and GP variance)
        median_sigma_eff = result["sigma_eff"].median()
        median_error = result["error"].median()
        
        assert median_sigma_eff >= median_error * 0.9  # Allow small tolerance


class TestRobustSigmaFloor:
    """Test the robust_sigma_floor estimation."""

    def test_sigma_floor_nonnegative(self):
        """sigma_floor should never produce negative values."""
        df = make_synthetic_lc()
        result = per_camera_gp_baseline(df, add_sigma_eff_col=True)
        
        # sigma_eff² = yerr² + floor² + var, so sigma_eff >= yerr
        # This indirectly tests floor >= 0
        assert (result["sigma_eff"] >= 0).all()

    def test_sigma_floor_robust_to_outliers(self):
        """sigma_floor should be stable when dips are present."""
        df_clean = make_synthetic_lc(seed=456)
        df_with_dip = inject_dip(df_clean, t0=df_clean["JD"].median(), amplitude=0.8)
        
        result_clean = per_camera_gp_baseline(df_clean, add_sigma_eff_col=True)
        result_dip = per_camera_gp_baseline(df_with_dip, add_sigma_eff_col=True)
        
        # Median sigma_eff shouldn't change dramatically due to one dip
        ratio = result_dip["sigma_eff"].median() / result_clean["sigma_eff"].median()
        assert 0.7 < ratio < 1.5, f"sigma_eff changed too much with dip: {ratio:.2f}"


class TestMaskedGPBehavior:
    """Test that masked GP properly excludes dips from fit."""

    def test_masked_baseline_stable_with_dip(self):
        """Masked GP baseline should follow quiescent level, not dip."""
        df = make_synthetic_lc(seed=789)
        t0 = df["JD"].median()
        df_dip = inject_dip(df, t0=t0, amplitude=1.0, sigma=5.0)
        
        result = per_camera_gp_baseline_masked(
            df_dip, 
            dip_sigma_thresh=-2.0,  # Aggressive masking
            add_sigma_eff_col=True
        )
        
        # Baseline at dip center should be close to quiescent level
        dip_mask = np.abs(df_dip["JD"] - t0) < 10
        baseline_at_dip = result.loc[dip_mask, "baseline"].mean()
        quiescent_mag = df["mag"].median()  # Original clean data
        
        # Baseline should be within 0.1 mag of quiescent level
        assert abs(baseline_at_dip - quiescent_mag) < 0.1


class TestBaselineFallbacks:
    """Test fallback behavior when GP fit fails."""

    def test_fallback_with_few_points(self):
        """Should fall back to median with insufficient points."""
        df = make_synthetic_lc(n_points=8, n_cameras=1)
        # per_camera_gp_baseline has internal min points check
        result = per_camera_gp_baseline(df, add_sigma_eff_col=True)
        
        assert "baseline" in result.columns
        assert result["baseline"].notna().all()

    def test_fallback_still_has_sigma_eff(self):
        """Fallback case should still compute sigma_eff."""
        df = make_synthetic_lc(n_points=8, n_cameras=1)
        result = per_camera_gp_baseline(df, add_sigma_eff_col=True)
        
        assert "sigma_eff" in result.columns
        assert result["sigma_eff"].notna().all()


class TestTrendBaseline:
    """Test per_camera_trend_baseline."""

    def test_trend_baseline_basic(self):
        """Trend baseline should produce expected columns."""
        df = make_synthetic_lc()
        result = per_camera_trend_baseline(df)
        
        assert "baseline" in result.columns
        assert "resid" in result.columns
        assert "sigma_resid" in result.columns

    def test_trend_baseline_sigma_eff_col(self):
        """Trend baseline should produce sigma_eff when requested."""
        df = make_synthetic_lc()
        result = per_camera_trend_baseline(df, add_sigma_eff_col=True)
        
        assert "sigma_eff" in result.columns
        assert result["sigma_eff"].notna().all()

    def test_trend_baseline_accepts_kwargs(self):
        """Trend baseline should accept extra kwargs for API compatibility."""
        df = make_synthetic_lc()
        # Should not raise even with GP-specific kwargs
        result = per_camera_trend_baseline(
            df, 
            add_sigma_eff_col=True,
            S0=0.001,  # GP param, should be ignored
            w0=0.01,   # GP param, should be ignored
        )
        assert "baseline" in result.columns


class TestEdgeCases:
    """Test edge cases in baseline computation."""

    def test_single_camera(self):
        """Handle single-camera light curves."""
        df = make_synthetic_lc(n_cameras=1)
        result = per_camera_gp_baseline(df, add_sigma_eff_col=True)
        
        assert len(result) == len(df)
        assert result["baseline"].notna().all()

    def test_some_nan_errors(self):
        """Handle light curves with some NaN errors."""
        df = make_synthetic_lc()
        df.loc[::3, "error"] = np.nan  # Every 3rd error is NaN
        
        result = per_camera_gp_baseline(df, add_sigma_eff_col=True)
        assert result["baseline"].notna().all()
        assert result["sigma_eff"].notna().all()

    def test_empty_dataframe(self):
        """Handle empty dataframe gracefully."""
        df = pd.DataFrame(columns=["JD", "mag", "error", "camera#", "saturated"])
        result = per_camera_gp_baseline(df, add_sigma_eff_col=True)
        
        assert len(result) == 0
