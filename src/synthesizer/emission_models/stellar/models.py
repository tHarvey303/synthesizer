"""A submodule containing the definitions of common stellar emission models.

This module contains the definitions of commoon stellar emission models that
can be used "out of the box" to generate spectra from components or as a
foundation to work from when creating more complex models.

Example usage::

    # Create a simple emission model
    model = TotalEmission(
        grid=grid,
        dust_curve=dust_curve,
        dust_emission_model=dust_emission_model,
        fesc=0.0,
    )

    # Generate the spectra
    spectra = stars.get_spectra(model)

"""

import numpy as np
from unyt import Angstrom

from synthesizer import exceptions
from synthesizer.emission_models.base_model import StellarEmissionModel
from synthesizer.emission_models.models import AttenuatedEmission, DustEmission
from synthesizer.emission_models.transformers import (
    EscapedFraction,
    ProcessedFraction,
)
from synthesizer.synth_warnings import warn


class IncidentEmission(StellarEmissionModel):
    """An emission model that extracts the incident radiation field.

    This defines an extraction of key "incident" from SPS grid.

    This is a child of the EmissionModel class for a full description of the
    parameters see the EmissionModel class.
    """

    def __init__(self, grid, label="incident", **kwargs):
        """Initialise the IncidentEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            label (str): The label for this emission model.
            **kwargs: Additional keyword arguments.
        """
        StellarEmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            extract="incident",
            **kwargs,
        )


class NebularLineEmission(StellarEmissionModel):
    """An emission model for the nebular line emission.

    This defines the luminosity contribution of the lines to the total
    nebular output.

    This is a child of the EmissionModel class; for a full description of the
    parameters see the EmissionModel class.
    """

    def __init__(
        self,
        grid,
        label="nebular_line",
        fesc_ly_alpha="fesc_ly_alpha",
        fesc="fesc",
        **kwargs,
    ):
        """Initialise the NebularLineEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            label (str): The label for this emission model.
            fesc_ly_alpha (float): The escape fraction of Lyman-alpha.
            fesc (float): The escape fraction of the emission.
            **kwargs: Additional keyword arguments.
        """
        # Get the lyman alpha wavelength elements and create a mask for
        # the line
        lyman_alpha_ind = np.argmin(np.abs(grid.lam - 1216.0 * Angstrom))
        lyman_alpha_mask = np.zeros(len(grid.lam), dtype=bool)
        lyman_alpha_mask[lyman_alpha_ind] = True

        # Since the spectra may have been resampled, we may have split the
        # lyman-alpha line into two. Therefore, we need to mask the
        # surrounding elements as well.
        if lyman_alpha_ind > 0:
            lyman_alpha_mask[lyman_alpha_ind - 1] = True
        if lyman_alpha_ind < len(grid.lam) - 1:
            lyman_alpha_mask[lyman_alpha_ind + 1] = True

        # For lyman-alpha, we reduce the overall luminosity by fesc,
        # then the remaining luminosity by fesc_ly_alpha
        lyman_alpha_no_fesc = StellarEmissionModel(
            label="_" + label + "_no_fesc",
            extract="linecont",
            grid=grid,
            save=False,
            **kwargs,
        )

        # We can't define fesc and fesc_ly_alpha in the same model if
        # the user tried tell them they can't
        if "fesc" in kwargs:
            raise exceptions.InconsistentArguments(
                "Cannot define fesc and fesc_ly_alpha in the same model. "
                "Please use another Transformation on NebularLineEmission "
                "to apply your fesc."
            )

        # Instantiate the combination model
        StellarEmissionModel.__init__(
            self,
            label=label,
            apply_to=lyman_alpha_no_fesc,
            transformer=EscapedFraction(fesc_attrs=("fesc_ly_alpha",)),
            fesc_ly_alpha=fesc_ly_alpha,
            lam_mask=lyman_alpha_mask,
            **kwargs,
        )


class TransmittedEmissionNoEscaped(StellarEmissionModel):
    """An emission model that extracts the transmitted radiation field.

    This defines an extraction of the key "transmitted" from. SPS grid.

    This is a child of the EmissionModel class for a full description of the
    parameters see the EmissionModel class.
    """

    def __init__(self, grid, label="transmitted", **kwargs):
        """Initialise the TransmittedEmissionNoEscaped object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            label (str): The label for this emission model.
            **kwargs: Additional keyword arguments.
        """
        StellarEmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            extract="transmitted",
            **kwargs,
        )


class TransmittedEmissionWithEscaped(StellarEmissionModel):
    """An emission model that extracts the transmitted radiation field.

    This defines 3 models:
      - An extraction of the key "transmitted" from SPS grid.
      - A transformed emission, for the transmitted radiation field accounting
        for the escape fraction.
      - A transformed emission, for the escaped radiation field accounting for
        the escape fraction.

    If fesc = 0.0 then there will only be the extraction model.

    This is a child of the EmissionModel class for a full description of the
    parameters see the EmissionModel class.
    """

    def __init__(
        self,
        grid,
        label="transmitted",
        fesc="fesc",
        related_models=(),
        incident=None,
        escaped_label="escaped",
        **kwargs,
    ):
        """Initialise the TransmittedEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            label (str): The label for this emission model.
            fesc (float): The escape fraction of the emission.
            related_models (list): A list of related models to combine with.
                This is used to combine the escaped and transmitted emission
                models.
            incident (EmissionModel): An incident emission model to use, if
                None then one will be created. This is only matters if
                fesc > 0.0, otherwise the incident contribution is 0.0.
            escaped_label (str): The label for the escaped emission model.
            **kwargs: Additional keyword arguments.
        """
        # Define the transmitted extraction model
        full_transmitted = StellarEmissionModel(
            grid=grid,
            label="full_" + label,
            extract="transmitted",
            **kwargs,
        )

        # We need an incident emission model to calculate the escaped if one
        # has not been passed warn the user we will make one
        if incident is None:
            warn(
                "TransmittedEmission requires an incident emission model. "
                f"We'll create one with the label '_{label}_incident'."
                " If you want to use a different incident model, please "
                "pass your own to the incident argument.",
            )
            incident = IncidentEmission(
                grid=grid,
                label=f"_{label}_incident",
                **kwargs,
            )

        # Get the escaped emission
        escaped = StellarEmissionModel(
            label=escaped_label,
            grid=grid,
            apply_to=incident,
            transformer=EscapedFraction(),
            fesc=fesc,
            **kwargs,
        )

        # Combine any extra related_models
        related_models = (escaped,) + tuple(related_models)

        # Get the transmitted emission (accounting for fesc)
        StellarEmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            apply_to=full_transmitted,
            transformer=ProcessedFraction(),
            fesc=fesc,
            related_models=related_models,
            **kwargs,
        )


class TransmittedEmission:
    """An emission model that extracts the transmitted radiation field.

    This is a wrapper around the TransmittedEmissionWithEscaped model
    and the TransmittedEmissionNoEscaped model. It will choose the
    appropriate model based on the inputs.

    This is a child of the EmissionModel class for a full description of the
    parameters see the EmissionModel class.
    """

    def __new__(
        cls,
        grid,
        label="transmitted",
        fesc="fesc",
        incident=None,
        related_models=(),
        escaped_label="escaped",
        **kwargs,
    ):
        """Initialise and return the correct TransmittedEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            label (str): The label for this emission model.
            fesc (float): The escape fraction of the emission, for no escaped
                emission this can either be set to 0.0 or None.
            incident (EmissionModel): An incident emission model to use, if
                None then one will be created. This is only matters if
                fesc > 0.0, otherwise the incident contribution is 0.0.
            related_models (list): A list of related models to combine with.
                This is used to combine the escaped and transmitted emission
                models.
            escaped_label (str): The label for the escaped emission model
                created if fesc > 0.0.
            **kwargs: Additional keyword arguments.
        """
        # If fesc is None or 0.0 then we only need the transmitted
        # emission without the escaped component.
        if fesc is None or fesc == 0.0:
            return TransmittedEmissionNoEscaped(
                grid=grid,
                label=label,
                related_models=related_models,
                **kwargs,
            )

        # Otherwise we need the transmitted emission with the escaped emission
        else:
            return TransmittedEmissionWithEscaped(
                grid=grid,
                label=label,
                fesc=fesc,
                incident=incident,
                related_models=related_models,
                escaped_label=escaped_label,
                **kwargs,
            )


class NebularContinuumEmission(StellarEmissionModel):
    """An emission model that extracts the nebular continuum emission.

    This defines an extraction of key "nebular_continuum" from. SPS grid.

    This is a child of the EmissionModel class for a full description of the
    parameters see the EmissionModel class .
    """

    def __init__(
        self,
        grid,
        label="nebular_continuum",
        **kwargs,
    ):
        """Initialise the NebularContinuumEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            label (str): The label for this emission model.
            **kwargs: Additional keyword arguments.
        """
        StellarEmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            extract="nebular_continuum",
            **kwargs,
        )


class NebularEmission(StellarEmissionModel):
    """An emission model that combines the nebular emissions.

    This defines a combination of the nebular continuum and line emission
    components.

    This is a child of the EmissionModel class for a full description of the
    parameters see the EmissionModel class .
    """

    def __init__(
        self,
        grid,
        label="nebular",
        fesc_ly_alpha="fesc_ly_alpha",
        nebular_line=None,
        nebular_continuum=None,
        **kwargs,
    ):
        """Initialise the NebularEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            label (str): The label for this emission model.
            fesc_ly_alpha (float): The escape fraction of Lyman-alpha.
            fesc (float): The escape fraction of the emission.
            nebular_line (EmissionModel): The nebular line model to use, if
                None then one will be created.
            nebular_continuum (EmissionModel): The nebular continuum model to
                use, if None then one will be created.
            **kwargs: Additional keyword arguments.
        """
        # If we have a Lyman-alpha escape fraction then calculate the
        # updated line emission and combine with the nebular continuum.
        # Make a nebular line model if we need one
        if nebular_line is None:
            warn(
                "NebularEmission requires a nebular line model. "
                f"We'll create one for you with the label '_{label}_line'. "
                "If you want to use a different nebular line model, please "
                "pass your own to the nebular_line argument.",
            )
            nebular_line = NebularLineEmission(
                grid=grid,
                fesc_ly_alpha=fesc_ly_alpha,
                label="_" + label + "_line",
                **kwargs,
            )

        # Make a nebular continuum model if we need one
        if nebular_continuum is None:
            warn(
                "NebularEmission requires a nebular continuum model. "
                "We'll create one for you with the label "
                f"'_{label}_continuum'. If you want to use a "
                "different nebular continuum model, please "
                "pass your own to the nebular_continuum argument.",
            )
            nebular_continuum = NebularContinuumEmission(
                grid=grid,
                label="_" + label + "_continuum",
                **kwargs,
            )

        StellarEmissionModel.__init__(
            self,
            label=label,
            combine=(nebular_line, nebular_continuum),
            **kwargs,
        )


class ReprocessedEmission(StellarEmissionModel):
    """An emission model that combines the reprocessed emission.

    This defines a combination of the nebular and transmitted components.

    This is a child of the EmissionModel class for a full description of the
    parameters see the EmissionModel class .
    """

    def __init__(
        self,
        grid,
        label="reprocessed",
        fesc="fesc",
        fesc_ly_alpha="fesc_ly_alpha",
        nebular=None,
        transmitted=None,
        **kwargs,
    ):
        """Initialise the ReprocessedEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            label (str): The label for this emission model.
            fesc (float): The escape fraction of the emission.
            fesc_ly_alpha (float): The escape fraction of Lyman-alpha.
            nebular (EmissionModel): The nebular model to use, if None then one
                will be created.
            transmitted (EmissionModel): The transmitted model to use, if None
                then one will be created.
            **kwargs: Additional keyword arguments.
        """
        # Make a nebular model if we need one
        if nebular is None:
            warn(
                "ReprocessedEmission requires a nebular model. "
                "We'll create one for you with the "
                f"label '_{label}_nebular'. If you want to use a "
                "different nebular model, please pass your own to the "
                "nebular argument.",
            )
            nebular = NebularEmission(
                grid=grid,
                label="_" + label + "_nebular",
                fesc_ly_alpha=fesc_ly_alpha,
                **kwargs,
            )

        # Make a transmitted model if we need one
        if transmitted is None:
            warn(
                "ReprocessedEmission requires a transmitted model. "
                "We'll create one for you with the label"
                f" '_{label}_transmitted'. If you want to use a "
                "different transmitted model, please pass your own to the "
                "transmitted argument.",
            )
            transmitted = TransmittedEmission(
                grid=grid,
                label="_" + label + "_transmitted",
                fesc=fesc,
                **kwargs,
            )

        StellarEmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            combine=(nebular, transmitted),
            **kwargs,
        )


class IntrinsicEmission:
    """An emission model that defines the intrinsic emission.

    This defines a combination of the reprocessed and escaped emission as
    long as we have an escape fraction greater than 0.0. Otherwise, it
    is identical to the reprocessed emission.

    This is a child of the EmissionModel class for a full description of the
    parameters see the EmissionModel class .
    """

    def __new__(
        cls,
        grid,
        label="intrinsic",
        fesc_ly_alpha="fesc_ly_alpha",
        fesc="fesc",
        reprocessed=None,
        **kwargs,
    ):
        """Initialise the IntrinsicEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            label (str): The label for this emission model.
            fesc_ly_alpha (float): The escape fraction of Lyman-alpha.
            fesc (float): The escape fraction of the emission.
            reprocessed (EmissionModel): The reprocessed model to use, if None
                then one will be created.
            escaped (EmissionModel): The escaped model to use, if None then one
                will be created. This is only used if fesc > 0.0.
            **kwargs: Additional keyword arguments.
        """
        # Make a reprocessed model if we need one
        if reprocessed is None:
            warn(
                "IntrinsicEmission requires a reprocessed model. "
                "We'll create one for you with the label"
                f" '_{label}_reprocessed'. If you want to use a "
                "different reprocessed model, please pass your own to the "
                "reprocessed argument.",
            )
            reprocessed = ReprocessedEmission(
                grid=grid,
                label="_" + label + "_reprocessed",
                fesc_ly_alpha=fesc_ly_alpha,
                fesc=fesc,
                **kwargs,
            )

        # If we have no escaped emission and no fesc then
        # intrinsic = reprocessed
        if fesc == 0.0 or fesc is None:
            warn(
                "IntrinsicEmission is identical to ReprocessedEmission when "
                "fesc is 0.0 or None. We'll return the reprocessed model "
                "instead of creating a new model.",
            )
            return reprocessed

        # Unpack the escaped emission from the reprocessed model
        escaped = reprocessed["escaped"]

        return StellarEmissionModel(
            grid=grid,
            label=label,
            combine=(escaped, reprocessed),
            **kwargs,
        )


class EmergentEmission(StellarEmissionModel):
    """An emission model that defines the emergent emission.

    This defines combination of the attenuated and escaped emission components
    to produce the emergent emission.

    This is a child of the EmissionModel class for a full description of the
    parameters see the EmissionModel class .
    """

    def __init__(
        self,
        grid,
        dust_curve=None,
        apply_to=None,
        fesc="fesc",
        fesc_ly_alpha="fesc_ly_alpha",
        label="emergent",
        attenuated=None,
        escaped=None,
        **kwargs,
    ):
        """Initialise the EmergentEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            dust_curve (AttenuationLaw): The dust curve to use.
            apply_to (EmissionModel): The model to apply the dust to.
            fesc (float): The escape fraction of the emission.
            fesc_ly_alpha (float): The escape fraction of Lyman-alpha.
            label (str): The label for this emission model.
            attenuated (EmissionModel): The attenuated model to use, if None
                then one will be created.
            escaped (EmissionModel): The escaped model to use, if None then one
                will be created.
            **kwargs: Additional keyword arguments.
        """
        # If apply_to is None then we need to make a model to apply to
        if apply_to is None and attenuated is None:
            warn(
                "EmergentEmission requires an apply_to model. "
                "We'll create one for you with the label "
                f"'_{label}_reprocessed'. If you want to apply dust to a "
                "different model, please pass your own to the "
                "apply_to argument.",
            )
            apply_to = ReprocessedEmission(
                grid=grid,
                label="_" + label + "_reprocessed",
                fesc=fesc,
                feac_ly_alpha=fesc_ly_alpha,
                **kwargs,
            )

        # Make an attenuated model if we need one
        if attenuated is None:
            warn(
                "EmergentEmission requires an attenuated model. "
                "We'll create one for you with the label "
                f"'_{label}_attenuated'. If you want to use a "
                "different attenuated model, please pass your own to the "
                "attenuated argument.",
            )
            attenuated = AttenuatedEmission(
                grid=grid,
                label="_" + label + "_attenuated",
                dust_curve=dust_curve,
                apply_to=apply_to,
                emitter="stellar",
                **kwargs,
            )

        # Do we have an escaped model?
        if escaped is None and "escaped" not in attenuated._models:
            raise exceptions.InconsistentArguments(
                "EmergentEmission requires an escaped model. "
                "Please pass your own to the escaped argument."
            )
        elif escaped is None:
            warn(
                "EmergentEmission requires an escaped model. "
                "We'll try to extract one from the attenuated model. "
                "If you want to use a different escaped model, please "
                "pass your own to the escaped argument.",
            )
            escaped = attenuated["escaped"]

        StellarEmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            combine=(attenuated, escaped),
            fesc=fesc,
            **kwargs,
        )


class TotalEmissionWithEscapedWithDust(StellarEmissionModel):
    """An emission model that defines total emission with an escape fraction.

    This defines the combination of the emergent and dust emission components
    to produce the total emission.

    This is a child of the EmissionModel class for a full description
    of the parameters see the EmissionModel class .
    """

    def __init__(
        self,
        grid,
        dust_curve,
        dust_emission_model,
        label="total",
        fesc="fesc",
        fesc_ly_alpha="fesc_ly_alpha",
        **kwargs,
    ):
        """Initialise the TotalEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            dust_curve (AttenuationLaw): The dust curve to use.
            dust_emission_model (synthesizer.dust.EmissionModel): The dust
                emission model to use.
            label (str): The label for this emission model.
            fesc (float): The escape fraction of the emission.
            fesc_ly_alpha (float): The escape fraction of Lyman-alpha.
            **kwargs: Additional keyword arguments.
        """
        # Set up models we need to link
        incident = IncidentEmission(
            grid=grid,
            label="incident",
            **kwargs,
        )
        nebular_line = NebularLineEmission(
            grid=grid,
            fesc_ly_alpha=fesc_ly_alpha,
            **kwargs,
        )
        nebular_continuum = NebularContinuumEmission(
            grid=grid,
            **kwargs,
        )
        nebular = NebularEmission(
            grid=grid,
            nebular_line=nebular_line,
            nebular_continuum=nebular_continuum,
            **kwargs,
        )
        transmitted = TransmittedEmission(
            grid=grid,
            fesc=fesc,
            incident=incident,
            **kwargs,
        )
        reprocessed = ReprocessedEmission(
            grid=grid,
            fesc=fesc,
            nebular=nebular,
            transmitted=transmitted,
            **kwargs,
        )
        attenuated = AttenuatedEmission(
            grid=grid,
            dust_curve=dust_curve,
            apply_to=reprocessed,
            emitter="stellar",
            **kwargs,
        )
        escaped = transmitted["escaped"]
        emergent = EmergentEmission(
            grid=grid,
            attenuated=attenuated,
            escaped=escaped,
            **kwargs,
        )
        dust_emission = DustEmission(
            dust_emission_model=dust_emission_model,
            dust_lum_intrinsic=reprocessed,
            dust_lum_attenuated=attenuated,
            emitter="stellar",
            **kwargs,
        )

        # Make the total emission model
        StellarEmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            combine=(
                emergent,
                dust_emission,
            ),
            **kwargs,
        )


class TotalEmissionNoEscapedWithDust(StellarEmissionModel):
    """An emission model that defines total emission.

    This defines the combination of the emergent and dust emission components
    to produce the total emission.

    This is a child of the EmissionModel class for a full description
    of the parameters see the EmissionModel class .
    """

    def __init__(
        self,
        grid,
        dust_curve,
        dust_emission_model,
        fesc_ly_alpha="fesc_ly_alpha",
        label="total",
        **kwargs,
    ):
        """Initialise the TotalEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            dust_curve (AttenuationLaw): The dust curve to use.
            dust_emission_model (synthesizer.dust.EmissionModel): The dust
                emission model to use.
            fesc_ly_alpha (float): The escape fraction of Lyman-alpha.
            label (str): The label for this emission model.
            **kwargs: Additional keyword arguments.
        """
        # Set up models we need to link
        incident = IncidentEmission(
            grid=grid,
            label="incident",
            **kwargs,
        )
        nebular_line = NebularLineEmission(
            grid=grid,
            fesc_ly_alpha=fesc_ly_alpha,
            **kwargs,
        )
        nebular_continuum = NebularContinuumEmission(
            grid=grid,
            **kwargs,
        )
        nebular = NebularEmission(
            grid=grid,
            nebular_line=nebular_line,
            nebular_continuum=nebular_continuum,
            **kwargs,
        )
        transmitted = TransmittedEmission(
            grid=grid,
            incident=incident,
            fesc=0.0,
            **kwargs,
        )
        reprocessed = ReprocessedEmission(
            grid=grid,
            nebular=nebular,
            transmitted=transmitted,
            **kwargs,
        )
        attenuated = AttenuatedEmission(
            grid=grid,
            dust_curve=dust_curve,
            apply_to=reprocessed,
            emitter="stellar",
            **kwargs,
        )
        dust_emission = DustEmission(
            dust_emission_model=dust_emission_model,
            dust_lum_intrinsic=reprocessed,
            dust_lum_attenuated=attenuated,
            emitter="stellar",
            **kwargs,
        )

        # Make the total emission model
        StellarEmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            combine=(
                attenuated,
                dust_emission,
            ),
            **kwargs,
        )


class TotalEmissionNoEscapedNoDust:
    """An emission model that defines total emission without dust emission.

    When no escape fraction is applied and no dust emission is included
    the total emission is simply the attenuated emission. This is just a
    helpful wrapper around that case.

    This is a child of the EmissionModel class for a full description
    of the parameters see the EmissionModel class .
    """

    def __new__(
        cls,
        grid,
        dust_curve,
        label="attenuated",
        fesc_ly_alpha="fesc_ly_alpha",
        **kwargs,
    ):
        """Initialise the TotalEmissionNoEscapeNoDust object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            dust_curve (AttenuationLaw): The dust curve to use.
            label (str): The label for this emission model.
            fesc_ly_alpha (float): The escape fraction of Lyman-alpha.
            **kwargs: Additional keyword arguments.
        """
        # Set up models we need to link
        incident = IncidentEmission(
            grid=grid,
            label="incident",
            **kwargs,
        )
        nebular_line = NebularLineEmission(
            grid=grid,
            fesc_ly_alpha=fesc_ly_alpha,
            **kwargs,
        )
        nebular_continuum = NebularContinuumEmission(
            grid=grid,
            **kwargs,
        )
        nebular = NebularEmission(
            grid=grid,
            nebular_line=nebular_line,
            nebular_continuum=nebular_continuum,
            **kwargs,
        )
        transmitted = TransmittedEmission(
            grid=grid,
            incident=incident,
            fesc=0.0,
            **kwargs,
        )
        reprocessed = ReprocessedEmission(
            grid=grid,
            nebular=nebular,
            transmitted=transmitted,
            **kwargs,
        )
        attenuated = AttenuatedEmission(
            grid=grid,
            dust_curve=dust_curve,
            apply_to=reprocessed,
            emitter="stellar",
            **kwargs,
        )
        return attenuated


class TotalEmissionWithEscapedNoDust:
    """An emission model that defines total emission with an escape fraction.

    When there is an escape fraction applied but no dust emission is included
    the total emission is simply the emergent emission. This is just a
    helpful wrapper around that case.

    This is a child of the EmissionModel class for a full description
    of the parameters see the EmissionModel class .
    """

    def __new__(
        cls,
        grid,
        dust_curve,
        fesc="fesc",
        fesc_ly_alpha="fesc_ly_alpha",
        label="total",
        **kwargs,
    ):
        """Initialise the TotalEmissionWithEscapeNoDust object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            dust_curve (AttenuationLaw): The dust curve to use.
            fesc (float): The escape fraction of the emission.
            fesc_ly_alpha (float): The escape fraction of Lyman-alpha.
            label (str): The label for this emission model.
            **kwargs: Additional keyword arguments.
        """
        # Set up models we need to link
        incident = IncidentEmission(
            grid=grid,
            label="incident",
            **kwargs,
        )
        nebular_line = NebularLineEmission(
            grid=grid,
            fesc_ly_alpha=fesc_ly_alpha,
            **kwargs,
        )
        nebular_continuum = NebularContinuumEmission(
            grid=grid,
            **kwargs,
        )
        nebular = NebularEmission(
            grid=grid,
            nebular_line=nebular_line,
            nebular_continuum=nebular_continuum,
            **kwargs,
        )
        transmitted = TransmittedEmission(
            grid=grid,
            fesc=fesc,
            incident=incident,
            **kwargs,
        )
        reprocessed = ReprocessedEmission(
            grid=grid,
            fesc=fesc,
            nebular=nebular,
            transmitted=transmitted,
            **kwargs,
        )
        attenuated = AttenuatedEmission(
            grid=grid,
            dust_curve=dust_curve,
            apply_to=reprocessed,
            emitter="stellar",
            **kwargs,
        )
        escaped = transmitted["escaped"]
        emergent = EmergentEmission(
            grid=grid,
            attenuated=attenuated,
            escaped=escaped,
            **kwargs,
        )
        return emergent


class TotalEmission:
    """An emission model that defines the total emission.

    This is a wrapper around the TotalEmissionWithEscape and
    TotalEmissionNoEscape models. It will choose the appropriate model based on
    the inputs.

    This is a child of the EmissionModel class for a full description
    of the parameters see the EmissionModel class .
    """

    def __new__(
        cls,
        grid,
        dust_curve,
        dust_emission_model=None,
        fesc="fesc",
        fesc_ly_alpha="fesc_ly_alpha",
        label="total",
        **kwargs,
    ):
        """Initialise and return the correct TotalEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            dust_curve (AttenuationLaw): The dust curve to use.
            dust_emission_model (synthesizer.dust.EmissionModel): The dust
                emission model to use.
            fesc (float): The escape fraction of the emission.
            fesc_ly_alpha (float): The escape fraction of Lyman-alpha.
            label (str): The label for this emission model.
            **kwargs: Additional keyword arguments.
        """
        # If fesc is None or 0.0 then we only need the total emission without
        # the escaped component.
        if fesc is None or fesc == 0.0:
            # If we have no dust emission then we can just return the
            # attenuated emission
            if dust_emission_model is None:
                return TotalEmissionNoEscapedNoDust(
                    grid=grid,
                    dust_curve=dust_curve,
                    label=label,
                    fesc_ly_alpha=fesc_ly_alpha,
                    **kwargs,
                )
            else:
                return TotalEmissionNoEscapedWithDust(
                    grid=grid,
                    dust_curve=dust_curve,
                    dust_emission_model=dust_emission_model,
                    fesc_ly_alpha=fesc_ly_alpha,
                    label=label,
                    **kwargs,
                )

        # Otherwise we need the total emission with the escaped component
        else:
            # If we have no dust emission then we can just return the
            # emergent emission
            if dust_emission_model is None:
                return TotalEmissionWithEscapedNoDust(
                    grid=grid,
                    dust_curve=dust_curve,
                    fesc=fesc,
                    fesc_ly_alpha=fesc_ly_alpha,
                    label=label,
                    **kwargs,
                )
            else:
                # Otherwise we return the total emission with the escaped
                # component
                return TotalEmissionWithEscapedWithDust(
                    grid=grid,
                    dust_curve=dust_curve,
                    dust_emission_model=dust_emission_model,
                    fesc=fesc,
                    fesc_ly_alpha=fesc_ly_alpha,
                    label=label,
                    **kwargs,
                )
