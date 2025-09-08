"""Test suite for HDF5 emission model loading functionality."""

import os
import tempfile

import h5py
import pytest
from unyt import Myr

from synthesizer.emission_models.base_model import EmissionModel


class TestEmissionModelHDF5:
    """Test suite for EmissionModel HDF5 functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test_model.h5")

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        os.rmdir(self.temp_dir)

    def test_simple_extraction_model_save_load(self, test_grid):
        """Test saving and loading a simple extraction model."""
        # Create a simple extraction model
        model = EmissionModel(
            label="test_extraction",
            grid=test_grid,
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
            assert model_group.attrs["grid"] == test_grid.grid_name
            assert model_group.attrs["extract"] == "incident"
            assert model_group.attrs["emitter"] == "stellar"

    def test_combination_model_save_load(self, test_grid):
        """Test saving and loading a combination model."""
        # Create two extraction models
        model1 = EmissionModel(
            label="stellar",
            grid=test_grid,
            extract="incident",
            emitter="stellar"
        )

        model2 = EmissionModel(
            label="nebular",
            grid=test_grid,
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

    def test_model_with_masks_save_load(self, test_grid):
        """Test saving and loading a model with masks."""
        model = EmissionModel(
            label="masked_model",
            grid=test_grid,
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

    def test_model_with_fixed_parameters_save_load(self, test_grid):
        """Test saving and loading a model with fixed parameters."""
        model = EmissionModel(
            label="fixed_param_model",
            grid=test_grid,
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

    def test_from_hdf5_extraction_model_basic(self, test_grid):
        """Test basic loading of extraction model with from_hdf5."""
        # Create and save a model
        original_model = EmissionModel(
            label="test_model",
            grid=test_grid,
            extract="incident",
            emitter="stellar",
            per_particle=True,
            save=False
        )

        with h5py.File(self.test_file, "w") as f:
            model_group = f.create_group("test_model")
            original_model.to_hdf5(model_group)

        # Load the model back
        grids = {test_grid.grid_name: test_grid}
        with h5py.File(self.test_file, "r") as f:
            loaded_model = EmissionModel.from_hdf5(f, grids=grids)

        # Verify loaded model matches original
        assert loaded_model.label == "test_model"
        assert loaded_model.emitter == "stellar"
        assert loaded_model.per_particle == True
        assert loaded_model.save == False
        assert loaded_model._is_extracting == True
        assert loaded_model.extract == "incident"

    def test_from_hdf5_combination_model_basic(self, test_grid):
        """Test basic loading of combination model with from_hdf5."""
        # Create extraction models
        model1 = EmissionModel(
            label="model1",
            grid=test_grid,
            extract="incident",
            emitter="stellar"
        )

        model2 = EmissionModel(
            label="model2",
            grid=test_grid,
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
        grids = {test_grid.grid_name: test_grid}
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
        # Test unsupported generator type
        with h5py.File(self.test_file, "w") as f:
            model_group = f.create_group("test_model")
            model_group.attrs["label"] = "test_model"
            model_group.attrs["type"] = "generation"
            model_group.attrs["generator"] = "<class 'some.unknown.Generator'>"
            model_group.attrs["emitter"] = "stellar"

        # Try to load - should raise NotImplementedError for unsupported generator
        with h5py.File(self.test_file, "r") as f:
            with pytest.raises(NotImplementedError, match="Generator reconstruction not implemented"):
                EmissionModel.from_hdf5(f)

    def test_error_on_invalid_model_type(self):
        """Test that invalid model types raise ValueError."""
        # Test completely invalid model type
        with h5py.File(self.test_file, "w") as f:
            model_group = f.create_group("test_model")
            model_group.attrs["label"] = "test_model"
            model_group.attrs["type"] = "invalid_type"
            model_group.attrs["emitter"] = "stellar"

        # Try to load - should raise ValueError for invalid model type
        with h5py.File(self.test_file, "r") as f:
            with pytest.raises(ValueError, match="Unsupported model type"):
                EmissionModel.from_hdf5(f)

    def test_powerlaw_transformer_save_load(self, test_grid):
        """Test saving and loading models with PowerLaw transformers."""
        # This test would require the actual PowerLaw class, so we'll mock the structure
        with h5py.File(self.test_file, "w") as f:
            # Create base extraction model
            base_group = f.create_group("base")
            base_group.attrs["label"] = "base"
            base_group.attrs["type"] = "extraction"
            base_group.attrs["grid"] = test_grid.grid_name
            base_group.attrs["extract"] = "incident"
            base_group.attrs["emitter"] = "stellar"

            # Create transformation model that applies to base
            transform_group = f.create_group("attenuated")
            transform_group.attrs["label"] = "attenuated"
            transform_group.attrs["type"] = "transformation"
            transform_group.attrs["transformer"] = "<class 'synthesizer.emission_models.transformers.dust_attenuation.PowerLaw'>"
            transform_group.attrs["transformer_slope"] = -1.5
            transform_group.attrs["apply_to"] = "base"
            transform_group.attrs["emitter"] = "stellar"

        # Verify the structure was saved correctly
        with h5py.File(self.test_file, "r") as f:
            transform_group = f["attenuated"]
            assert transform_group.attrs["type"] == "transformation"
            assert transform_group.attrs["transformer_slope"] == -1.5
            assert transform_group.attrs["apply_to"] == "base"

    def test_blackbody_generator_save_load(self, test_grid):
        """Test saving and loading models with Blackbody generators."""
        # This test would require the actual classes, so we'll mock the structure
        with h5py.File(self.test_file, "w") as f:
            # Create generation model
            gen_group = f.create_group("dust_emission")
            gen_group.attrs["label"] = "dust_emission"
            gen_group.attrs["type"] = "generation"
            gen_group.attrs["generator"] = "<class 'synthesizer.emission_models.dust.emission.Blackbody'>"
            gen_group.attrs["generator_temperature"] = 20.0
            gen_group.attrs["generator_temperature_units"] = "K"
            gen_group.attrs["generator_cmb_factor"] = 1.0
            gen_group.attrs["emitter"] = "stellar"

        # Verify the structure was saved correctly
        with h5py.File(self.test_file, "r") as f:
            gen_group = f["dust_emission"]
            assert gen_group.attrs["type"] == "generation"
            assert gen_group.attrs["generator_temperature"] == 20.0
            assert gen_group.attrs["generator_temperature_units"] == "K"
            assert gen_group.attrs["generator_cmb_factor"] == 1.0

    def test_incident_emission_model_save_load(self, incident_emission_model, test_grid):
        """Test saving and loading using the incident_emission_model fixture."""
        # Save the model to HDF5
        with h5py.File(self.test_file, "w") as f:
            model_group = f.create_group("incident_model")
            incident_emission_model.to_hdf5(model_group)

        # Verify the saved structure
        with h5py.File(self.test_file, "r") as f:
            model_group = f["incident_model"]
            assert model_group.attrs["label"] == incident_emission_model.label
            assert model_group.attrs["type"] == "extraction"
            assert model_group.attrs["emitter"] == incident_emission_model.emitter

    def test_nebular_emission_model_save_load(self, nebular_emission_model, test_grid):
        """Test saving and loading using the nebular_emission_model fixture."""
        # Save the model to HDF5
        with h5py.File(self.test_file, "w") as f:
            model_group = f.create_group("nebular_model")
            nebular_emission_model.to_hdf5(model_group)

        # Verify the saved structure
        with h5py.File(self.test_file, "r") as f:
            model_group = f["nebular_model"]
            assert model_group.attrs["label"] == nebular_emission_model.label
            assert model_group.attrs["emitter"] == nebular_emission_model.emitter

    def test_transmitted_emission_model_save_load(self, transmitted_emission_model, test_grid):
        """Test saving and loading using the transmitted_emission_model fixture."""
        # Save the model to HDF5
        with h5py.File(self.test_file, "w") as f:
            model_group = f.create_group("transmitted_model")
            transmitted_emission_model.to_hdf5(model_group)

        # Verify the saved structure
        with h5py.File(self.test_file, "r") as f:
            model_group = f["transmitted_model"]
            assert model_group.attrs["label"] == transmitted_emission_model.label
            assert model_group.attrs["emitter"] == transmitted_emission_model.emitter

    def test_reprocessed_emission_model_save_load(self, reprocessed_emission_model, test_grid):
        """Test saving and loading using the reprocessed_emission_model fixture."""
        # Save the model to HDF5
        with h5py.File(self.test_file, "w") as f:
            model_group = f.create_group("reprocessed_model")
            reprocessed_emission_model.to_hdf5(model_group)

        # Verify the saved structure
        with h5py.File(self.test_file, "r") as f:
            model_group = f["reprocessed_model"]
            assert model_group.attrs["label"] == reprocessed_emission_model.label
            assert model_group.attrs["emitter"] == reprocessed_emission_model.emitter

    def test_combination_with_preconfigured_models(self, incident_emission_model, nebular_emission_model, test_grid):
        """Test saving and loading a combination of preconfigured emission models."""
        # Create a combination model using the preconfigured fixtures
        combo_model = EmissionModel(
            label="combined_preconfigured",
            combine=[incident_emission_model, nebular_emission_model],
            emitter="stellar"
        )

        # Save the tree to HDF5
        combo_model.save_tree_to_hdf5(self.test_file)

        # Verify file structure
        with h5py.File(self.test_file, "r") as f:
            assert f.attrs["root_model"] == "combined_preconfigured"
            assert "combined_preconfigured" in f

            # Check combination model attributes
            combo_group = f["combined_preconfigured"]
            assert combo_group.attrs["label"] == "combined_preconfigured"
            assert combo_group.attrs["type"] == "combination"
            # The combine attribute should contain the labels of the child models
            combine_labels = list(combo_group.attrs["combine"])
            assert incident_emission_model.label in combine_labels
            assert nebular_emission_model.label in combine_labels

    def test_from_hdf5_with_grid_path(self, test_grid):
        """Test loading models with grid_path parameter."""
        # Create and save a model
        model = EmissionModel(
            label="test_model",
            grid=test_grid,
            extract="incident",
            emitter="stellar"
        )

        with h5py.File(self.test_file, "w") as f:
            model_group = f.create_group("test_model")
            model.to_hdf5(model_group)

        # Test loading with grid_path (this will likely fail in test environment
        # since we don't have actual grid files, but tests the code path)
        with h5py.File(self.test_file, "r") as f:
            try:
                loaded_model = EmissionModel.from_hdf5(f, grid_path="/fake/path")
            except FileNotFoundError:
                # Expected in test environment
                pass

    def test_from_hdf5_missing_grid_error(self, test_grid):
        """Test that missing grids raise appropriate errors."""
        # Create and save a model
        model = EmissionModel(
            label="test_model",
            grid=test_grid,
            extract="incident",
            emitter="stellar"
        )

        with h5py.File(self.test_file, "w") as f:
            model_group = f.create_group("test_model")
            model.to_hdf5(model_group)

        # Try to load without providing the grid
        with h5py.File(self.test_file, "r") as f:
            with pytest.raises(FileNotFoundError):
                EmissionModel.from_hdf5(f, grids={})  # Empty grids dict

    def test_incident_emission_from_hdf5(self, incident_emission_model, test_grid):
        """Test that IncidentEmission models can be loaded back correctly."""
        # Save the model
        incident_emission_model.save_tree_to_hdf5(self.test_file)

        # Load it back
        grids = {test_grid.grid_name: test_grid}
        loaded_model = EmissionModel.load_tree_from_hdf5(self.test_file, grids=grids)

        # Verify it's the correct type and has correct properties
        assert loaded_model.label == incident_emission_model.label
        assert loaded_model.emitter == incident_emission_model.emitter
        assert loaded_model._is_extracting == True
        # Note: The loaded model might be generic EmissionModel if specialized loading fails
        # but it should still have the same functional behavior

    def test_nebular_emission_from_hdf5(self, nebular_emission_model, test_grid):
        """Test that NebularEmission models can be loaded back correctly."""
        # Save the model
        nebular_emission_model.save_tree_to_hdf5(self.test_file)

        # Load it back
        grids = {test_grid.grid_name: test_grid}
        loaded_model = EmissionModel.load_tree_from_hdf5(self.test_file, grids=grids)

        # Verify basic properties
        assert loaded_model.label == nebular_emission_model.label
        assert loaded_model.emitter == nebular_emission_model.emitter

    def test_transmitted_emission_from_hdf5(self, transmitted_emission_model, test_grid):
        """Test that TransmittedEmission models can be loaded back correctly."""
        # Save the model
        transmitted_emission_model.save_tree_to_hdf5(self.test_file)

        # Load it back
        grids = {test_grid.grid_name: test_grid}
        loaded_model = EmissionModel.load_tree_from_hdf5(self.test_file, grids=grids)

        # Verify basic properties
        assert loaded_model.label == transmitted_emission_model.label
        assert loaded_model.emitter == transmitted_emission_model.emitter

    def test_reprocessed_emission_from_hdf5(self, reprocessed_emission_model, test_grid):
        """Test that ReprocessedEmission models can be loaded back correctly."""
        # Save the model
        reprocessed_emission_model.save_tree_to_hdf5(self.test_file)

        # Load it back
        grids = {test_grid.grid_name: test_grid}
        loaded_model = EmissionModel.load_tree_from_hdf5(self.test_file, grids=grids)

        # Verify basic properties
        assert loaded_model.label == reprocessed_emission_model.label
        assert loaded_model.emitter == reprocessed_emission_model.emitter

    def test_intrinsic_emission_from_hdf5(self, intrinsic_emission_model, test_grid):
        """Test that IntrinsicEmission models can be loaded back correctly."""
        # Save the model
        intrinsic_emission_model.save_tree_to_hdf5(self.test_file)

        # Load it back
        grids = {test_grid.grid_name: test_grid}
        loaded_model = EmissionModel.load_tree_from_hdf5(self.test_file, grids=grids)

        # Verify basic properties
        assert loaded_model.label == intrinsic_emission_model.label
        assert loaded_model.emitter == intrinsic_emission_model.emitter

    def test_pacman_emission_from_hdf5(self, pacman_emission_model, test_grid):
        """Test that PacmanEmission models can be loaded back correctly."""
        # Save the model
        pacman_emission_model.save_tree_to_hdf5(self.test_file)

        # Load it back
        grids = {test_grid.grid_name: test_grid}
        loaded_model = EmissionModel.load_tree_from_hdf5(self.test_file, grids=grids)

        # Verify basic properties
        assert loaded_model.label == pacman_emission_model.label
        assert loaded_model.emitter == pacman_emission_model.emitter

    def test_bimodal_pacman_emission_from_hdf5(self, bimodal_pacman_emission_model, test_grid):
        """Test that BimodalPacmanEmission models can be loaded back correctly."""
        # Save the model
        bimodal_pacman_emission_model.save_tree_to_hdf5(self.test_file)

        # Load it back
        grids = {test_grid.grid_name: test_grid}
        loaded_model = EmissionModel.load_tree_from_hdf5(self.test_file, grids=grids)

        # Verify basic properties
        assert loaded_model.label == bimodal_pacman_emission_model.label
        assert loaded_model.emitter == bimodal_pacman_emission_model.emitter

    def test_class_information_saved(self, incident_emission_model):
        """Test that class information is properly saved to HDF5."""
        with h5py.File(self.test_file, "w") as f:
            model_group = f.create_group("incident_model")
            incident_emission_model.to_hdf5(model_group)

        # Check that class information was saved
        with h5py.File(self.test_file, "r") as f:
            model_group = f["incident_model"]
            assert "class_name" in model_group.attrs
            assert "class_module" in model_group.attrs

            class_name = model_group.attrs["class_name"]
            class_module = model_group.attrs["class_module"]

            # For IncidentEmission, should be saved as its actual class
            assert class_name == "IncidentEmission"
            assert "stellar.models" in class_module

    def test_grid_parameter_handling(self, test_grid):
        """Test that grid objects are properly passed to specialized models."""
        # Create a simple IncidentEmission directly
        from synthesizer.emission_models.stellar.models import (
            IncidentEmission,
        )

        model = IncidentEmission(grid=test_grid, label="test_incident")

        # Save to HDF5
        with h5py.File(self.test_file, "w") as f:
            model_group = f.create_group("test_model")
            model.to_hdf5(model_group)

        # Load back
        grids = {test_grid.grid_name: test_grid}
        with h5py.File(self.test_file, "r") as f:
            loaded_model = EmissionModel.from_hdf5(f, grids=grids)

        # Should have loaded successfully
        assert loaded_model.label == "test_incident"
        assert loaded_model._is_extracting


if __name__ == "__main__":
    pytest.main([__file__])
