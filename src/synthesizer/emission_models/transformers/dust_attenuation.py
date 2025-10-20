"""A module containing dust attenuation functionality.

This module contains classes for dust attenuation laws. These classes
provide a way to calculate the optical depth and transmission curves
for a given dust model.

Example usage::

    # Create a power law dust model with a slope of -1.0
    dust_model = PowerLaw(slope=-1.0)

    # Calculate the transmission curve
    transmission = dust_model.get_transmission(
        0.33, np.linspace(1000, 10000, 1000)
    )
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from dust_extinction.grain_models import WD01
from scipy import interpolate
from unyt import angstrom, um

from synthesizer import exceptions
from synthesizer.emission_models.transformers.transformer import Transformer
from synthesizer.units import accepts

this_dir, this_filename = os.path.split(__file__)

__all__ = ["PowerLaw", "MWN18", "Calzetti2000", "GrainsWD01", "ParametricLi08"]

_RESET_SENTINEL = object()


class AttenuationLaw(Transformer):
    """The base class for all attenuation laws.

    A child of this class should define it's own get_tau method with any
    model specific behaviours. This will be used by get_transmission (which
    itself can be overloaded by the child if needed).

    Attributes:
        description (str):
            A description of the type of model. Defined on children classes.
        required_params (tuple):
            The name of any required parameters needed by the transformer
            when transforming an emission. These should either be
            available from an emitter or from the EmissionModel itself.
            If they are missing an exception will be raised.
    """

    def __init__(self, description, required_params=("tau_v",)):
        """Initialise the parent and set common attributes.

        Args:
            description (str):
                A description of the type of model.
            required_params (tuple):
                List of required model attributes.
        """
        # Store the description of the model.
        self.description = description
        # Store user-supplied conversions between model arguments and parameter
        # names, e.g. allows the user to set e.g. slope = 'slope_young' on the
        # emitter or model and have that passed to the dust curve as slope
        self._name_transforms = {}
        # Stores overridden parameters temporarily
        self._temp_params = {}
        if "tau_v" not in required_params:
            raise exceptions.InconsistentArguments(
                "AttenuationLaw requires 'tau_v' as a parameter."
            )
        # Call the parent constructor
        Transformer.__init__(self, required_params=required_params)

    def get_tau(self, *args):
        """Compute the V-band normalised optical depth."""
        raise exceptions.UnimplementedFunctionality(
            "AttenuationLaw should not be instantiated directly!"
            " Instead use one to child models (" + ", ".join(__all__) + ")"
        )

    def get_tau_at_lam(self, *args):
        """Compute the optical depth at wavelength."""
        raise exceptions.UnimplementedFunctionality(
            "AttenuationLaw should not be instantiated directly!"
            " Instead use one to child models (" + ", ".join(__all__) + ")"
        )

    @accepts(lam=angstrom)
    def get_transmission(self, tau_v, lam, **dust_curve_kwargs):
        """Compute the transmission curve.

        Returns the transmitted flux/luminosity fraction based on an optical
        depth at a range of wavelengths.

        Args:
            tau_v (float/np.ndarray of float):
                Optical depth in the V-band. Can either be a single float or
                array.
            lam (np.ndarray of float):
                The wavelengths (with units) at which to calculate
                transmission.
            **dust_curve_kwargs (dict):
                Additional keyword arguments to be passed to the dust curve
                which have been defined on the emitter or model.

        Returns:
            np.ndarray of float:
                The transmission at each wavelength. Either (lam.size,) in
                shape for singular tau_v values or (tau_v.size, lam.size)
                tau_v is an array.
        """
        # Set any additional parameters on the dust curve
        self._set_params(**dust_curve_kwargs)

        try:
            # Get the optical depth at each wavelength
            tau_x_v = self.get_tau(lam)
        finally:
            # Always restore previous state
            self._reset_params()

        # Include the V band optical depth in the exponent
        # For a scalar we can just multiply but for an array we need to
        # broadcast
        if np.isscalar(tau_v):
            exponent = tau_v * tau_x_v
        else:
            if np.ndim(lam) == 0:
                exponent = tau_v[:] * tau_x_v
            else:
                exponent = tau_v[:, None] * tau_x_v

        return np.exp(-exponent)

    def _check_required_params(self):
        """Get the required parameters for the transformer.

        Given the input params, this method will return the required
        params by looking at required params which have been
        set as strings or None.

        """
        param_values = [
            getattr(self, param, None) for param in self._required_params
        ]

        required_params = []
        for param, value in zip(self._required_params, param_values):
            if value is None:
                required_params.append(param)
            elif isinstance(value, str):
                required_params.append(value)
                self._name_transforms[value] = param

        self._required_params = required_params

    def _transform(self, emission, emitter, model, mask, lam_mask):
        """Apply the dust attenuation to the emission.

        Args:
            emission (Line/Sed): The emission to transform.
            emitter (Stars/Gas/BlackHole/Galaxy): The object emitting the
                emission.
            model (EmissionModel): The emission model generating the emission.
            mask (np.ndarray): The mask to apply to the emission.
            lam_mask (np.ndarray): We must define this parameter in the
                transformer method, but it is not used in this case. If not
                None an error will be raised.

        Returns:
            Line/Sed: The transformed emission.
        """
        # Extract the required parameters
        params = self._extract_params(model, emission, emitter)

        # Ensure we aren't trying to use a wavelength mask
        if lam_mask is not None:
            raise exceptions.UnimplementedFunctionality(
                "Wavelength mask currently not supported in dust attenuation."
            )

        # Apply the transmission to the emission
        return emission.apply_attenuation(
            dust_curve=self,
            mask=mask,
            **params,
        )

    def _set_params(self, **params):
        """Set the parameters of the dust curve.

        This method will set any parameters defined in params as attributes
        of the dust curve. This allows for parameters to be set on the
        emitter or model and then used by the dust curve.

        Args:
            **params (dict):
                The parameters to set.
        """
        # Save existing state of only the attributes we will override,
        # then apply overrides mapped to the actual attribute names.
        self._temp_params = {}
        overrides = {}
        for key, value in params.items():
            attr = self._name_transforms.get(key, key)
            overrides[attr] = value
        for attr, value in overrides.items():
            prev = getattr(self, attr, _RESET_SENTINEL)
            self._temp_params[attr] = prev
            setattr(self, attr, value)

    def _reset_params(self):
        """Reset the parameters of the dust curve to their previous state."""
        for attr, prev in self._temp_params.items():
            if prev is _RESET_SENTINEL:
                # Attribute did not exist prior to override
                if hasattr(self, attr):
                    delattr(self, attr)
            else:
                setattr(self, attr, prev)
        self._temp_params = {}

    @accepts(lam=angstrom)
    def plot_attenuation(
        self,
        lam,
        fig=None,
        ax=None,
        label=None,
        figsize=(8, 6),
        show=True,
        **kwargs,
    ):
        """Plot the attenuation curve.

        Args:
            lam (np.ndarray of float):
                The wavelengths (with units) at which to calculate
                transmission.
            fig (matplotlib.figure.Figure):
                The figure to plot on. If None, a new figure will be created.
            ax (matplotlib.axes.Axes):
                The axis to plot on. If None, a new axis will be created.
            label (str):
                The label to use for the plot.
            figsize (tuple):
                The size of the figure to create if fig is None.
            show (bool):
                Whether to show the plot.
            **kwargs (dict):
                Keyword arguments to be provided to the `plot` call.

        Returns:
            fig, ax:
                The figure and axis objects.
        """
        if fig is None:
            fig = plt.figure(figsize=figsize)
        if ax is None:
            ax = fig.add_subplot(111)

        # Get the attenuation curve
        a_V = self.get_tau(lam)

        # Plot the transmission curve
        ax.plot(lam, a_V, label=label, **kwargs)

        # Add labels
        ax.set_xlabel(r"$\lambda/(\AA)$")
        ax.set_ylabel(r"A$_{\lambda}/$A$_{V}$")

        ax.set_yticks(np.arange(0, 10))
        ax.set_xlim(np.min(lam), np.max(lam))
        ax.set_ylim(0.0, 10)

        # Add a legend if the ax has labels to plot
        if any(ax.get_legend_handles_labels()[1]):
            ax.legend()

        # Show the plot
        if show:
            plt.show()

        return fig, ax

    @accepts(lam=angstrom)
    def plot_transmission(
        self,
        tau_v,
        lam,
        fig=None,
        ax=None,
        label=None,
        figsize=(8, 6),
        show=True,
    ):
        """Plot the transmission curve.

        Args:
            tau_v (float/np.ndarray of float):
                Optical depth in the V-band. Can either be a single float or
                array.
            lam (np.ndarray of float):
                The wavelengths (with units) at which to calculate
                transmission.
            fig (matplotlib.figure.Figure):
                The figure to plot on. If None, a new figure will be created.
            ax (matplotlib.axes.Axes):
                The axis to plot on. If None, a new axis will be created.
            label (str):
                The label to use for the plot.
            figsize (tuple):
                The size of the figure to create if fig is None.
            show (bool):
                Whether to show the plot.

        Returns:
            fig, ax:
                The figure and axis objects.
        """
        if fig is None:
            fig = plt.figure(figsize=figsize)
        if ax is None:
            ax = fig.add_subplot(111)

        # Get the transmission curve
        transmission = self.get_transmission(tau_v, lam)

        # Plot the transmission curve
        ax.plot(lam, transmission, label=label)

        # Add labels
        ax.set_xlabel("Wavelength (Angstrom)")
        ax.set_ylabel("Transmission")

        # Add a legend if the ax has labels to plot
        if any(ax.get_legend_handles_labels()[1]):
            ax.legend()

        # Show the plot
        if show:
            plt.show()

        return fig, ax


class PowerLaw(AttenuationLaw):
    """Custom power law dust curve.

    Attributes:
        slope (float):
            The slope of the power law.
    """

    def __init__(self, slope=-1.0):
        """Initialise the power law slope of the dust curve.

        Args:
            slope (float):
                The slope of the power law dust curve.
        """
        description = "simple power law dust curve"
        AttenuationLaw.__init__(
            self, description, required_params=("tau_v", "slope")
        )
        self.slope = slope

        self._check_required_params()

    @accepts(lam=angstrom)
    def get_tau_at_lam(self, lam):
        """Calculate optical depth at a wavelength.

        Args:
            lam (float/np.ndarray of float):
                An array of wavelengths or a single wavlength at which to
                calculate optical depths.

        Returns:
            float/np.ndarray of float: The optical depth.
        """
        return (lam / (5500.0 * angstrom)) ** self.slope

    @accepts(lam=angstrom)
    def get_tau(self, lam):
        """Calculate V-band normalised optical depth.

        Args:
            lam (float/np.ndarray of float):
                An array of wavelengths or a single wavlength at which to
                calculate optical depths.

        Returns:
            float/np.ndarray of float: The optical depth.
        """
        return self.get_tau_at_lam(lam) / self.get_tau_at_lam(
            5500.0 * angstrom
        )


@accepts(lam=angstrom, cent_lam=angstrom, gamma=angstrom)
def N09Tau(lam, slope, cent_lam, ampl, gamma):
    """Generate the transmission curve for the Noll+2009 attenuation curve.

    Attenuation curve using a modified version of the Calzetti
    attenuation (Calzetti+2000) law allowing for a varying UV slope
    and the presence of a UV bump; from Noll+2009

    References:
        https://ui.adsabs.harvard.edu/abs/2009A%26A...499...69N

    Args:
        lam (np.ndarray of float):
            The input wavelength array (expected in AA units,
            global unit).
        slope (float):
            The slope of the attenuation curve.
        cent_lam (float):
            The central wavelength of the UV bump, expected in AA.
        ampl (float):
            The amplitude of the UV-bump.
        gamma (float):
            The width (FWHM) of the UV bump, in AA.

    Returns:
        np.ndarray of float: V-band normalised optical depth for
            given wavelength
    """
    # Performing some unit conversions to match the
    # Calzetti curve units which are in um
    _lam = np.linspace(0.01, 3.0, 10000, endpoint=True) * um
    _cent_lam = cent_lam.to("um")
    _gamma = gamma.to("um")
    lam_v = 0.55  # in um

    k_lam = np.zeros_like(_lam.value)

    # Masking for different regimes in the Calzetti curve
    ok1 = (_lam >= 0.12) * (_lam < 0.63)  # 0.12um<=lam<0.63um
    ok2 = (_lam >= 0.63) * (_lam < 3.1)  # 0.63um<=lam<=3.10um
    ok3 = _lam < 0.12  # lam<0.12um
    if np.sum(ok1) > 0:  # equation 1
        k_lam[ok1] = (
            -2.156
            + (1.509 / _lam.value[ok1])
            - (0.198 / _lam.value[ok1] ** 2)
            + (0.011 / _lam.value[ok1] ** 3)
        )
        func = interpolate.interp1d(
            _lam.value[ok1], k_lam[ok1], fill_value="extrapolate"
        )
    else:
        func = None
    if np.sum(ok2) > 0:  # equation 2
        k_lam[ok2] = -1.857 + (1.040 / _lam.value[ok2])
    if np.sum(ok3) > 0:
        # This will never be none
        if func is None:
            raise exceptions.InconsistentArguments("No data in the UV-optical")
        # Extrapolating the 0.12um<=lam<0.63um regime
        k_lam[ok3] = func(_lam.value[ok3])

    # Using the Calzetti attenuation curve normalised
    # to Av=4.05
    k_lam = 4.05 + 2.659 * k_lam
    k_v = 4.05 + 2.659 * (
        -2.156 + (1.509 / lam_v) - (0.198 / lam_v**2) + (0.011 / lam_v**3)
    )

    # UV bump feature expression from Noll+2009
    D_lam = (
        ampl
        * ((_lam * _gamma) ** 2)
        / ((_lam**2 - _cent_lam**2) ** 2 + (_lam * _gamma) ** 2)
    )

    # Normalising with the value at 0.55um, to obtain
    # normalised optical depth
    tau_x_v = (k_lam + D_lam) / k_v
    tau_x = tau_x_v * (_lam.value / lam_v) ** slope

    func = interpolate.interp1d(
        _lam, tau_x, bounds_error=False, fill_value=(tau_x[0], tau_x[-1])
    )

    return func(lam.to("um"))


class Calzetti2000(AttenuationLaw):
    """Calzetti attenuation curve.

    This includes options for the slope and UV-bump implemented in
    Noll et al. 2009.

    Attributes:
        slope (float):
            The slope of the attenuation curve.

        cent_lam (float):
            The central wavelength of the UV bump, expected in AA.

        ampl (float):
            The amplitude of the UV-bump.

        gamma (float):
            The width (FWHM) of the UV bump, in AA.

    """

    @accepts(cent_lam=angstrom, gamma=angstrom)
    def __init__(
        self,
        slope=0,
        cent_lam=2175 * angstrom,
        ampl=0,
        gamma=350 * angstrom,
    ):
        """Initialise the dust curve.

        Args:
            slope (float):
                The slope of the attenuation curve.

            cent_lam (float):
                The central wavelength of the UV bump, expected in AA.

            ampl (float):
                The amplitude of the UV-bump.

            gamma (float):
                The width (FWHM) of the UV bump, in AA.
        """
        description = (
            "Calzetti attenuation curve; with option"
            "for the slope and UV-bump implemented"
            "in Noll et al. 2009"
        )

        required_params = ("tau_v", "slope", "cent_lam", "ampl", "gamma")

        AttenuationLaw.__init__(self, description, required_params)

        # Define the parameters of the model.
        self.slope = slope
        self.cent_lam = cent_lam
        self.ampl = ampl
        self.gamma = gamma

        self._check_required_params()

    @accepts(lam=angstrom)
    def get_tau(self, lam):
        """Calculate V-band normalised optical depth.

        (Uses the N09Tau function defined above.)

        Args:
            lam (float/np.ndarray of float):
                An array of wavelengths or a single wavlength at which to
                calculate optical depths (in AA, global unit).

        Returns:
            float/array-like, float:
                The V-band noramlised optical depth.
        """
        return N09Tau(
            lam=lam,
            slope=self.slope,
            cent_lam=self.cent_lam,
            ampl=self.ampl,
            gamma=self.gamma,
        )

    @accepts(lam=angstrom)
    def get_tau_at_lam(self, lam):
        """Calculate optical depth at a wavelength.

        (Uses the N09Tau function defined above.)

        Args:
            lam (float/array-like, float):
                An array of wavelengths or a single wavlength at which to
                calculate optical depths (in AA, global unit).

        Returns:
            float/array-like, float
                The optical depth.
        """
        tau_x_v = N09Tau(
            lam=lam,
            slope=self.slope,
            cent_lam=self.cent_lam,
            ampl=self.ampl,
            gamma=self.gamma,
        )

        # V-band wavelength in micron
        # Since N09Tau uses micron for calculations
        lam_v = 0.55

        k_v = 4.05 + 2.659 * (
            -2.156 + (1.509 / lam_v) - (0.198 / lam_v**2) + (0.011 / lam_v**3)
        )

        return k_v * tau_x_v


class MWN18(AttenuationLaw):
    """Milky Way attenuation curve used in Narayanan+2018.

    Attributes:
        data (np.ndarray of float):
            The data describing the dust curve, loaded from MW_N18.npz.
        tau_lam_v (float):
            The V band optical depth.
    """

    def __init__(self):
        """Initialise the dust curve.

        This will load the data and get the V band optical depth by
        interpolation.
        """
        description = "MW extinction curve from Desika"
        AttenuationLaw.__init__(self, description)
        self.data = np.load(f"{this_dir}/../../data/MW_N18.npz")
        self.tau_lam_v = np.interp(
            5500.0, self.data.f.mw_df_lam[::-1], self.data.f.mw_df_chi[::-1]
        )

    @accepts(lam=angstrom)
    def get_tau(self, lam, interp="cubic"):
        """Calculate V-band normalised optical depth.

        Args:
            lam (float/array, float):
                An array of wavelengths or a single wavlength at which to
                calculate optical depths (in AA, global unit).
            interp (str):
                The type of interpolation to use. Can be 'linear', 'nearest',
                'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic',
                'previous', or 'next'. 'zero', 'slinear', 'quadratic' and
                'cubic' refer to a spline interpolation of zeroth, first,
                second or third order. Uses scipy.interpolate.interp1d.

        Returns:
            float/array, float: The optical depth.
        """
        func = interpolate.interp1d(
            self.data.f.mw_df_lam[::-1],
            self.data.f.mw_df_chi[::-1],
            kind=interp,
            fill_value="extrapolate",
        )
        return func(lam) / self.tau_lam_v

    @accepts(lam=angstrom)
    def get_tau_at_lam(self, lam, interp="cubic"):
        """Calculate the optical depth at a wavelength.

        Args:
            lam (float/array, float):
                An array of wavelengths or a single wavlength at which to
                calculate optical depths (in AA, global unit).
            interp (str):
                The type of interpolation to use. Can be 'linear', 'nearest',
                'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic',
                'previous', or 'next'. 'zero', 'slinear', 'quadratic' and
                'cubic' refer to a spline interpolation of zeroth, first,
                second or third order. Uses scipy.interpolate.interp1d.

        Returns:
            float/array, float
                The optical depth.
        """
        func = interpolate.interp1d(
            self.data.f.mw_df_lam[::-1],
            self.data.f.mw_df_chi[::-1],
            kind=interp,
            fill_value="extrapolate",
        )
        return func(lam)


class GrainsWD01(AttenuationLaw):
    """Weingarter and Draine 2001 dust grain extinction model.

    Includes curves for MW, SMC and LMC or any available in WD01.

    Attributes:
        model (str):
            The dust grain model used.
        emodel (function)
            The function that describes the model from WD01 imported above.
    """

    def __init__(self, model="SMCBar"):
        """Initialise the dust curve.

        Args:
            model (str):
                The dust grain model to use.
        """
        AttenuationLaw.__init__(
            self,
            "Weingarter and Draine 2001 dust grain model for MW, SMC, and LMC",
        )

        # Get the correct model string
        if "MW" in model:
            self.model = "MWRV31"
        elif "LMC" in model:
            self.model = "LMCAvg"
        elif "SMC" in model:
            self.model = "SMCBar"
        else:
            self.model = model
        self.emodel = WD01(self.model)

    @accepts(lam=angstrom)
    def get_tau(self, lam, interp="slinear"):
        """Calculate V-band normalised optical depth.

        Args:
            lam (float/np.ndarray of float):
                An array of wavelengths or a single wavlength at which to
                calculate optical depths (in AA, global unit).

            interp (str):
                The type of interpolation to use. Can be 'linear', 'nearest',
                'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic',
                'previous', or 'next'. 'zero', 'slinear', 'quadratic' and
                'cubic' refer to a spline interpolation of zeroth, first,
                second or third order. Uses scipy.interpolate.interp1d.

        Returns:
            float/np.ndarray of float: The optical depth.
        """
        lam_lims = np.logspace(2, 8, 10000) * angstrom
        lam_v = 5500 * angstrom  # V-band wavelength
        func = interpolate.interp1d(
            lam_lims,
            self.emodel(lam_lims.to_astropy()),
            kind=interp,
            fill_value="extrapolate",
        )
        out = func(lam) / func(lam_v)

        if np.isscalar(lam):
            if lam > lam_lims[-1]:
                out = func(lam_lims[-1])
        elif np.sum(lam > lam_lims[-1]) > 0:
            out[(lam > lam_lims[-1])] = func(lam_lims[-1])

        return out

    @accepts(lam=angstrom)
    def get_tau_at_lam(self, lam, interp="slinear"):
        """Calculate optical depth at a wavelength.

        Args:
            lam (float/array-like, float):
                An array of wavelengths or a single wavlength at which to
                calculate optical depths (in AA, global unit).
            interp (str):
                The type of interpolation to use. Can be 'linear', 'nearest',
                'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic',
                'previous', or 'next'. 'zero', 'slinear', 'quadratic' and
                'cubic' refer to a spline interpolation of zeroth, first,
                second or third order. Uses scipy.interpolate.interp1d.

        Returns:
            float/array-like, float
                The optical depth.
        """
        lam_lims = np.logspace(2, 8, 10000) * angstrom
        func = interpolate.interp1d(
            lam_lims,
            self.emodel(lam_lims.to_astropy()),
            kind=interp,
            fill_value="extrapolate",
        )
        out = func(lam)

        if np.isscalar(lam):
            if lam > lam_lims[-1]:
                out = func(lam_lims[-1])
        elif np.sum(lam > lam_lims[-1]) > 0:
            out[(lam > lam_lims[-1])] = func(lam_lims[-1])

        return out


@accepts(lam=angstrom)
def Li08(lam, UV_slope, OPT_NIR_slope, FUV_slope, bump, model):
    """Drude-like parametric expression for the attenuation curve from Li+08.

    Args:
        lam (np.ndarray of float):
            The wavelengths (AA units) at which to calculate transmission.
        UV_slope (float):
            Dimensionless parameter describing the UV-FUV slope
        OPT_NIR_slope (float):
            Dimensionless parameter describing the optical/NIR slope
        FUV_slope (float):
            Dimensionless parameter describing the FUV slope
        bump (float):
            Dimensionless parameter describing the UV bump
            strength (0< bump <1)
        model (str):
            Via this parameter one can choose one of the templates for
            extinction/attenuation curves from: Calzetti, SMC, MW (R_V=3.1),
            and LMC

    Returns:
            np.ndarray of float/float: tau/tau_v at each input wavelength (lam)
    """
    # Empirical templates (Calzetti, SMC, MW RV=3.1, LMC)
    if model == "Calzetti":
        UV_slope, OPT_NIR_slope, FUV_slope, bump = 44.9, 7.56, 61.2, 0.0
    if model == "SMC":
        UV_slope, OPT_NIR_slope, FUV_slope, bump = 38.7, 3.83, 6.34, 0.0
    if model == "MW":
        UV_slope, OPT_NIR_slope, FUV_slope, bump = 14.4, 6.52, 2.04, 0.0519
    if model == "LMC":
        UV_slope, OPT_NIR_slope, FUV_slope, bump = 4.47, 2.39, -0.988, 0.0221

    # Converting lam from AA to um for ease
    _lam = lam.to("um")

    # Attenuation curve (normalized to Av)
    term1 = UV_slope / (
        (_lam.value / 0.08) ** OPT_NIR_slope
        + (_lam.value / 0.08) ** -OPT_NIR_slope
        + FUV_slope
    )
    term2 = (
        233.0
        * (
            1
            - UV_slope
            / (6.88**OPT_NIR_slope + 0.145**OPT_NIR_slope + FUV_slope)
            - bump / 4.6
        )
    ) / ((_lam.value / 0.046) ** 2.0 + (_lam.value / 0.046) ** -2.0 + 90.0)
    term3 = bump / (
        (_lam.value / 0.2175) ** 2.0 + (_lam.value / 0.2175) ** -2.0 - 1.95
    )

    AlamAV = term1 + term2 + term3

    return AlamAV


class ParametricLi08(AttenuationLaw):
    """Parametric, empirical attenuation curve.

    Implemented in Li+08, Evolution of the parameters up to high-z
    (z=12) studied in: Markov+23a,b

    Attributes:
        UV_slope (float):
            Dimensionless parameter describing the UV-FUV slope (0, 50)
        OPT_NIR_slope (float):
            Dimensionless parameter describing the optical/NIR slope (0, 10)
        FUV_slope (float):
            Dimensionless parameter describing the FUV slope (-1, 75)
        bump (float):
            Dimensionless parameter describing the UV bump
            strength (-0.005< bump <0.06)
        model (str):
            Fixing attenuation/extinction curve to one of the known
            templates: MW, SMC, LMC, Calzetti

    """

    def __init__(
        self,
        UV_slope=44.9,
        OPT_NIR_slope=7.56,
        FUV_slope=61.2,
        bump=0.0,
        model="Calzetti",
    ):
        """Initialise the dust curve.

        Args:
            UV_slope (float):
                Dimensionless parameter describing the UV-FUV slope (0, 50)
            OPT_NIR_slope (float):
                Dimensionless parameter describing the optical/NIR
                slope (0, 10)
            FUV_slope (float):
                Dimensionless parameter describing the FUV slope (-1, 75)
            bump (float):
                Dimensionless parameter describing the UV bump
                strength (-0.005< bump <0.06)
            model (str):
                Fixing attenuation/extinction curve to one of the known
                templates: MW, SMC, LMC, Calzetti
        """
        description = (
            "Parametric attenuation curve; with option"
            "for multiple slopes (UV_slope,OPT_NIR_slope,FUV_slope) and "
            "varying UV-bump strength (bump)."
            "Introduced in Li+08, see Markov+23,24 for application to "
            "a high-z dataset."
            "Empirical extinction/attenuation curves (MW, SMC, LMC, "
            "Calzetti) can be selected."
        )

        required_params = (
            "tau_v",
            "UV_slope",
            "OPT_NIR_slope",
            "FUV_slope",
            "bump",
        )

        AttenuationLaw.__init__(self, description, required_params)

        # Define the parameters of the model.
        self.UV_slope = UV_slope
        self.OPT_NIR_slope = OPT_NIR_slope
        self.FUV_slope = FUV_slope
        self.bump = bump

        # Get the correct model string
        if model == "MW":
            self.model = "MW"
        elif model == "LMC":
            self.model = "LMC"
        elif model == "SMC":
            self.model = "SMC"
        elif model == "Calzetti":
            self.model = "Calzetti"
        else:
            self.model = "Custom"

        self._check_required_params()

    @accepts(lam=angstrom)
    def get_tau(self, lam):
        """Calculate V-band normalised optical depth.

        (Uses the Li_08 function defined above.)

        Args:
            lam (float/np.ndarray of float):
                An array of wavelengths or a single wavlength at which to
                calculate optical depths (in AA, global unit).

        Returns:
            float/array-like, float: The V-band normalised optical depth.
        """
        return Li08(
            lam=lam,
            UV_slope=self.UV_slope,
            OPT_NIR_slope=self.OPT_NIR_slope,
            FUV_slope=self.FUV_slope,
            bump=self.bump,
            model=self.model,
        )

    @accepts(lam=angstrom)
    def get_tau_at_lam(self, lam):
        """Calculate optical depth at a wavelength.

        (Uses the Li_08 function defined above.)

        Args:
            lam (float/array-like, float):
                An array of wavelengths or a single wavlength at which to
                calculate optical depths (in AA, global unit).

        Returns:
            float/array-like, float
                The optical depth.
        """
        raise exceptions.UnimplementedFunctionality(
            "ParametricLi08 form is fit to the normalised Alam/Av values"
            "for the different models, so does not make sense to have this"
            "function. Use other attenuation curve models to get tau_v or Av"
        )
