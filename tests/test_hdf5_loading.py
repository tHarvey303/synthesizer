"""Test suite for HDF5 emission model loading functionality."""

import tempfile
import os
import pytest
import h5py
import numpy as np
from unyt import K, Myr

from synthesizer.emission_models.base_model import EmissionModel
from synthesizer.grid import Grid


class MockGrid:
    """Mock grid class for testing without needing actual grid files."""
    
    def __init__(self, grid_name="test_grid"):
        self.grid_name = grid_name
        self.available_emissions = ["incident", "transmitted", "nebular"]
        self.lam = np.logspace(2, 5, 1000)  # Wavelength array
        
    def __getattr__(self, name):
        # Return mock values for any grid attribute
        if name == "available_spectra_emissions":
            return self.available_emissions
        elif name == "available_line_emissions":
            return []
        return None


class TestEmissionModelHDF5:
    """Test suite for EmissionModel HDF5 functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test_model.h5")
        
        # Create a mock grid
        self.mock_grid = MockGrid("test_grid")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        os.rmdir(self.temp_dir)
    
    def test_simple_extraction_model_save_load(self):
        """Test saving and loading a simple extraction model."""
        # Create a simple extraction model
        model = EmissionModel(
            label="test_extraction",
            grid=self.mock_grid,
            extract="incident",
            emitter="stellar"
        )
        
        # Save to HDF5
        with h5py.File(self.test_file, "w") as f:
            model_group = f.create_group("model")
            model.to_hdf5(model_group)
        
        # Verify file was created and has expected structure
        with h5py.File(self.test_file, "r") as f:
            assert "model" in f
            model_group = f["model"]
            
            # Check basic attributes
            assert model_group.attrs["label"] == "test_extraction"
            assert model_group.attrs["type"] == "extraction"
            assert model_group.attrs["grid"] == "test_grid"
            assert model_group.attrs["extract"] == "incident"
            assert model_group.attrs["emitter"] == "stellar"
    
    def test_combination_model_save_load(self):
        """Test saving and loading a combination model."""
        # Create two extraction models
        model1 = EmissionModel(
            label="stellar",
            grid=self.mock_grid,
            extract="incident",
            emitter="stellar"
        )
        
        model2 = EmissionModel(
            label="nebular", 
            grid=self.mock_grid,
            extract="nebular",
            emitter="stellar"
        )
        
        # Create combination model
        combo_model = EmissionModel(
            label="combined",
            combine=[model1, model2],
            emitter="stellar"
        )
        
        # Save the tree to HDF5
        combo_model.save_tree_to_hdf5(self.test_file)
        
        # Verify file structure
        with h5py.File(self.test_file, "r") as f:
            assert f.attrs["root_model"] == "combined"
            assert "combined" in f
            assert "stellar" in f
            assert "nebular" in f
            
            # Check combination model attributes
            combo_group = f["combined"]
            assert combo_group.attrs["label"] == "combined"
            assert combo_group.attrs["type"] == "combination"
            assert list(combo_group.attrs["combine"]) == ["stellar", "nebular"]
    
    def test_model_with_masks_save_load(self):
        """Test saving and loading a model with masks."""
        model = EmissionModel(
            label="masked_model",
            grid=self.mock_grid,
            extract="incident",
            emitter="stellar",
            mask_attr="age",
            mask_thresh=100 * Myr,
            mask_op=">"
        )
        
        # Save to HDF5
        with h5py.File(self.test_file, "w") as f:
            model_group = f.create_group("model")
            model.to_hdf5(model_group)
        
        # Verify mask information was saved
        with h5py.File(self.test_file, "r") as f:
            model_group = f["model"]
            assert "Masks" in model_group
            masks_group = model_group["Masks"]
            assert "mask_0" in masks_group
            
            mask_group = masks_group["mask_0"]
            assert mask_group.attrs["attr"] == "age"
            assert mask_group.attrs["op"] == ">"
            assert mask_group.attrs["thresh"] == 100
            assert mask_group.attrs["thresh_units"] == "Myr"
    
    def test_model_with_fixed_parameters_save_load(self):
        """Test saving and loading a model with fixed parameters."""
        model = EmissionModel(
            label="fixed_param_model",
            grid=self.mock_grid,
            extract="incident",
            emitter="stellar",
            tau_v=0.5,
            fesc=0.1
        )
        
        # Save to HDF5
        with h5py.File(self.test_file, "w") as f:
            model_group = f.create_group("model")
            model.to_hdf5(model_group)
        
        # Verify fixed parameters were saved
        with h5py.File(self.test_file, "r") as f:
            model_group = f["model"]
            assert "FixedParameters" in model_group
            fixed_params = model_group["FixedParameters"]
            assert fixed_params.attrs["tau_v"] == 0.5
            assert fixed_params.attrs["fesc"] == 0.1
    
    def test_from_hdf5_extraction_model_basic(self):
        """Test basic loading of extraction model with from_hdf5."""
        # Create and save a model
        original_model = EmissionModel(
            label="test_model",
            grid=self.mock_grid,
            extract="incident",
            emitter="stellar",
            per_particle=True,
            save=False
        )
        
        with h5py.File(self.test_file, "w") as f:
            model_group = f.create_group("test_model")
            original_model.to_hdf5(model_group)
        
        # Load the model back
        grids = {"test_grid": self.mock_grid}
        with h5py.File(self.test_file, "r") as f:
            loaded_model = EmissionModel.from_hdf5(f, grids=grids)
        
        # Verify loaded model matches original
        assert loaded_model.label == "test_model"
        assert loaded_model.emitter == "stellar" 
        assert loaded_model.per_particle == True
        assert loaded_model.save == False
        assert loaded_model._is_extracting == True
        assert loaded_model.extract == "incident"
    
    def test_from_hdf5_combination_model_basic(self):
        """Test basic loading of combination model with from_hdf5."""
        # Create extraction models
        model1 = EmissionModel(
            label="model1",
            grid=self.mock_grid,
            extract="incident", 
            emitter="stellar"
        )
        
        model2 = EmissionModel(
            label="model2",
            grid=self.mock_grid,
            extract="nebular",
            emitter="stellar"
        )
        
        # Create combination
        combo_model = EmissionModel(
            label="combo",
            combine=[model1, model2],
            emitter="stellar"
        )
        
        # Save tree
        combo_model.save_tree_to_hdf5(self.test_file)
        
        # Load back
        grids = {"test_grid": self.mock_grid}
        loaded_model = EmissionModel.load_tree_from_hdf5(self.test_file, grids=grids)
        
        # Verify structure
        assert loaded_model.label == "combo"
        assert loaded_model._is_combining == True
        assert len(loaded_model.combine) == 2
        
        # Check that child models were reconstructed
        child_labels = [child.label for child in loaded_model.combine]
        assert "model1" in child_labels
        assert "model2" in child_labels
    
    def test_error_on_unsupported_model_type(self):
        """Test that unsupported model types raise appropriate errors."""
        # Create a mock HDF5 structure for an unsupported model type
        with h5py.File(self.test_file, "w") as f:
            model_group = f.create_group("test_model")
            model_group.attrs["label"] = "test_model"
            model_group.attrs["type"] = "transformation"  # Not yet supported
            model_group.attrs["emitter"] = "stellar"
        
        # Try to load - should raise NotImplementedError
        with h5py.File(self.test_file, "r") as f:
            with pytest.raises(NotImplementedError, match="transformation.*not yet fully implemented"):
                EmissionModel.from_hdf5(f)


if __name__ == "__main__":
    pytest.main([__file__])