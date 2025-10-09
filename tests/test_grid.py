"""A test suite for the Grid class.

TODO: This needs to actually be implemented with proper useful tests.
"""

import os
from pathlib import Path

import numpy as np

from synthesizer.grid import Grid


def test_grid_returned(test_grid):
    """Test that a Grid object is returned."""
    assert isinstance(test_grid, Grid)


def test_grid_axes(test_grid):
    """Test that the axes are returned correctly."""
    # Test the automatic extraction defined by the __getattr__ overload
    assert getattr(test_grid.ages, "units", None) is not None
    assert getattr(test_grid.metallicities, "units", None) is not None
    assert isinstance(test_grid._ages, np.ndarray)
    assert np.allclose(test_grid._ages, 10**test_grid.log10ages)

    # Test the silly singular and plural nonsense works (hopefully we can
    # remove this in the future)
    assert np.allclose(test_grid._ages, test_grid._age)
    assert np.allclose(test_grid._metallicities, test_grid._metallicity)
    assert np.allclose(test_grid.log10ages, test_grid.log10age)


class TestSPSGridLines:
    """Tests for SPS lines grids."""

    def test_lines_different(self, test_grid):
        """Test that the lines are different."""
        # Ensure that all values aren't the same for the whole grid
        luminosities = test_grid.line_lums["nebular"]
        conts = test_grid.line_conts["nebular"]
        assert not np.unique(luminosities).size == 1, (
            f"All line luminosities are the same {luminosities.min()}"
        )
        assert not np.unique(conts).size == 1, (
            f"All line conts are the same {conts.min()}"
        )

        # Ensure that none of the lines are all the same
        non_unique_lum_lines = []
        non_unique_cont_lines = []
        non_unique_lum_vals = []
        non_unique_cont_vals = []
        for ind, line in enumerate(test_grid.available_lines):
            # Skip entirely zeroed lines, these have meaning
            if np.sum(luminosities[..., ind]) == 0:
                continue
            if np.unique(luminosities[..., ind]).size == 1:
                non_unique_lum_lines.append(line)
                non_unique_lum_vals.append(luminosities[..., ind].min())

            if np.unique(conts[..., ind]).size == 1:
                non_unique_cont_lines.append(line)
                non_unique_cont_vals.append(conts[..., ind].min())

        assert non_unique_lum_lines == [], (
            f"{non_unique_lum_lines} found with constant luminosity "
            f"values in the grid ({non_unique_lum_vals})"
        )
        assert non_unique_cont_lines == [], (
            f"{non_unique_cont_lines} found with constant continuum "
            f"values in the grid ({non_unique_cont_vals})"
        )


def test_convert_to_bagpipes(test_grid, tmp_path):
    """Test that the convert_to_bagpipes method creates expected files."""
    # Create a temporary output directory
    output_dir = tmp_path / "bagpipes_output"

    # Run the conversion
    output_files = test_grid.convert_to_bagpipes(
        output_dir=str(output_dir),
        logU=-2.0
    )

    # Check that the output directory was created
    assert output_dir.exists()

    # Check that the stellar file was created
    assert "stellar" in output_files
    stellar_file = Path(output_files["stellar"])
    assert stellar_file.exists()
    assert stellar_file.suffix == ".fits"

    # Check that nebular files were created (if grid has lines)
    if test_grid.reprocessed and test_grid.available_lines:
        assert "nebular_line" in output_files
        assert "nebular_cont" in output_files
        assert "cloudy_lines" in output_files
        assert "cloudy_linewavs" in output_files

        line_file = Path(output_files["nebular_line"])
        cont_file = Path(output_files["nebular_cont"])
        cloudy_lines_file = Path(output_files["cloudy_lines"])
        cloudy_linewavs_file = Path(output_files["cloudy_linewavs"])

        assert line_file.exists()
        assert cont_file.exists()
        assert cloudy_lines_file.exists()
        assert cloudy_linewavs_file.exists()

        # Check file extensions
        assert line_file.suffix == ".fits"
        assert cont_file.suffix == ".fits"
        assert cloudy_lines_file.suffix == ".txt"
        assert cloudy_linewavs_file.suffix == ".txt"

