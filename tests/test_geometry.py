"""Tests for manywells.geometry.WellGeometry."""

import pytest
import numpy as np

from manywells.geometry import WellGeometry


def test_well_geometry_invalid_D():
    """WellGeometry rejects non-positive diameter."""
    with pytest.raises(ValueError, match="diameter"):
        WellGeometry.vertical(2000, 100, D=0)


def test_well_geometry_cos_incl_validation():
    """WellGeometry rejects cos_incl outside [0, 1] (TVD decreasing)."""
    with pytest.raises(ValueError, match="cos_incl"):
        WellGeometry(md_survey=[0, 100, 200], tvd_survey=[0, 100, 80])


class TestWellGeometry:
    """Tests for the WellGeometry dataclass."""

    def test_vertical_geometry(self):
        """Vertical well: MD == TVD, cos_incl == 1 everywhere."""
        geo = WellGeometry.vertical(2000, 100)
        assert geo.L == 2000
        assert geo.n_cells == 100
        assert len(geo.md) == 101
        assert len(geo.tvd) == 101
        assert len(geo.cos_incl) == 100
        assert all(c == pytest.approx(1.0) for c in geo.cos_incl)

    def test_deviated_geometry(self):
        """Deviated well: cos_incl < 1 in deviated sections."""
        md = [0, 500, 1000, 2000]
        tvd = [0, 500, 800, 1200]
        geo = WellGeometry.from_survey(md_survey=md, tvd_survey=tvd, n_cells=10)
        assert geo.L == 2000
        assert any(c < 0.99 for c in geo.cos_incl)

    def test_md_geq_tvd_validation(self):
        """MD must be >= TVD at every survey station."""
        with pytest.raises(ValueError, match="MD must be >= TVD"):
            WellGeometry(md_survey=[0, 100], tvd_survey=[0, 200])

    def test_non_monotonic_md_rejected(self):
        """Non-monotonic MD is rejected."""
        with pytest.raises(ValueError, match="strictly increasing"):
            WellGeometry(md_survey=[0, 100, 50], tvd_survey=[0, 100, 50])

    def test_origin_validation(self):
        """Survey must start at (0, 0)."""
        with pytest.raises(ValueError, match="start at 0"):
            WellGeometry(md_survey=[10, 100], tvd_survey=[10, 100])

    def test_frozen(self):
        """WellGeometry instances are immutable."""
        geo = WellGeometry.vertical(1000, 10)
        with pytest.raises(AttributeError):
            geo.D = 0.2

    def test_tvd_frac_bounds(self):
        """tvd_frac is 1.0 at the bottom (index 0) and 0.0 at the surface (last index)."""
        geo = WellGeometry.vertical(2000, 50)
        assert geo.tvd_frac[0] == pytest.approx(1.0)
        assert geo.tvd_frac[-1] == pytest.approx(0.0)

    def test_simulator_order(self):
        """md[0] is the deepest point, md[-1] is the surface."""
        geo = WellGeometry.vertical(2000, 10)
        assert geo.md[0] == pytest.approx(2000.0)
        assert geo.md[-1] == pytest.approx(0.0)
        assert geo.tvd[0] == pytest.approx(2000.0)
        assert geo.tvd[-1] == pytest.approx(0.0)

    def test_l_shaped_geometry(self):
        """L-shaped well: 2000 m vertical, gradual build, 1000 m horizontal."""
        R = 250.0  # build-section radius of curvature (m)

        # Survey: vertical -> circular arc (vertical to horizontal) -> horizontal
        md_survey = [0.0, 2000.0]
        tvd_survey = [0.0, 2000.0]

        n_arc = 5
        theta = np.linspace(0, np.pi / 2, n_arc + 1)[1:]
        for t in theta:
            md_survey.append(2000.0 + R * t)
            tvd_survey.append(2000.0 + R * np.sin(t))

        md_toe = md_survey[-1] + 1000.0
        tvd_toe = tvd_survey[-1]
        md_survey.append(md_toe)
        tvd_survey.append(tvd_toe)

        n_cells = 100
        geo = WellGeometry.from_survey(md_survey=md_survey, tvd_survey=tvd_survey, n_cells=n_cells)

        assert geo.L == pytest.approx(md_toe)
        assert len(geo.cos_incl) == n_cells
        assert len(geo.md) == n_cells + 1

        # Simulator order: index 0 = toe, index -1 = surface
        assert geo.md[0] == pytest.approx(md_toe)
        assert geo.md[-1] == pytest.approx(0.0)
        assert geo.tvd[0] == pytest.approx(2000.0 + R)
        assert geo.tvd[-1] == pytest.approx(0.0)

        # cos_incl in vertical section (near-surface cells = high index) ~ 1
        for c in geo.cos_incl[-20:]:
            assert c == pytest.approx(1.0, abs=0.01)

        # cos_incl in horizontal section (near-toe cells = low index) ~ 0
        for c in geo.cos_incl[:20]:
            assert c == pytest.approx(0.0, abs=0.01)

        # All inclinations are physically valid
        assert all(0 - 1e-9 <= c <= 1 + 1e-9 for c in geo.cos_incl)
