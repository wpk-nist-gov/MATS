MATS Summary
============

In multi-spectrum fitting, there are a collection of spectra modeled by the same line-by-line spectroscopic parameters, but each spectrum might vary in pressure, temperature, and sample composition.

The MATS program is based on :py:class:`MATS.Spectrum` objects, which are defined not only by their wavenumber and absorption data, but also information on the spectrum pressure, temperature, baseline characteristics, and sample composition.  In addition to utilizing real spectra, MATS has a :py:func:`MATS.simulate_spectrum` function, which returns a spectrum object calculated from input simulation parameters.  This is useful for performing error analysis in the same framework as primary data analysis by simply switching from experimental to simulated :py:class:`MATS.Spectrum` objects.  These :py:class:`MATS.Spectrum` objects are combined to form a :py:class:`MATS.Dataset` object, which is the collection of spectra that are being analyzed together in the multi-spectrum analysis.

There are two files that contain parameters that are fit in this model, one for spectrum dependent parameters (polynomial baseline parameters, etalons, sample composition, and x-shift term) and the other for line-by-line spectroscopic parameters that are common across all spectra.  These files are saved as .csv files with a column for each parameter and with rows corresponding to either the spectrum number or spectral line number.  In addition to the columns containing the values for the fit parameters, there are two additional columns for each fittable parameter called param_vary and param_err.  The param_vary column is a boolean  (True/False) flag that is toggled to indicate whether a given parameter will be varied in the fit.  The param_err column will be set to zero initially and replaced with the standard error for the parameter determined by the fit results.  Calls of the :py:class:`MATS.Generate_FitParam_File` class not only make these input files, but also set the line shape and define whether a parameter should be varied in the fit and if a parameter should be constrained across all spectra or allowed to vary by spectrum.  The :py:class:`MATS.Edit_Fit_Param_Files` class allows for edits to the parameter files in the Jupyter Notebook interface rather than editing in the .csv file.

Finally, the :py:class:`MATS.Fit_DataSet` class fits the spectra.  Additionally, it allows the user to impose constraints on the parameters (min and max values), impose convergence criteria, update background and parameter line lists, and plot fit results.

Below is the sparse documentation for each of the classes and main functions in the MATS project with links to the full documentation provided.

Spectrum Class and Objects
++++++++++++++++++++++++++

.. currentmodule:: MATS.Spectrum



.. autosummary::
   MATS.Spectrum
   MATS.simulate_spectrum
   MATS.Spectrum.calculate_QF
   MATS.Spectrum.fft_spectrum
   MATS.Spectrum.plot_freq_tau
   MATS.Spectrum.plot_model_residuals
   MATS.Spectrum.plot_wave_alpha
   MATS.Spectrum.save_spectrum_info
   MATS.Spectrum.segment_wave_alpha




Line-by-line Model and Etalon Models
++++++++++++++++++++++++++++++++++++
The line-by-line model is based on the HTP code provided in the `HITRAN Application Programming Interface (HAPI) <https://hitran.org/hapi/>`_.  For the most part the conventions and definitions used by HITRAN are used in the MATS program.  However, for some of the advanced line profile parameters the naming convention and temperature dependence is different.  In the sections below, the temperature and pressure dependence of the various parameters is outlined for clarity.

The Hartmann-Tran profile limiting cases that correspond to several commonly used line profiles.  These limiting cases are achieved by setting line shape parameters equal to 0. The list below indicates the parameters that are not fixed equal to zero in each of the HTP limiting case line shapes.  For more information about the HTP see the following references: `Recommended isolated-line profile for representing high-resolution spectroscopic transitions (IUPAC Technical Report) <https://www.degruyter.com/view/journals/pac/86/12/article-p1931.xml>`_ and `An isolated line-shape model to go beyond the Voigt profile in spectroscopic databases and radiative transfer codes <https://www.sciencedirect.com/science/article/pii/S0022407313002422>`_


Voigt Profile (VP):  :math:`\Gamma_{D}, \Gamma_{0}, \Delta_{0}`

Nelkin-Ghatak Profile (NGP):  :math:`\Gamma_{D}, \Gamma_{0}, \Delta_{0}, \nu_{VC}`

speed-dependent Voigt Profile (SDVP):  :math:`\Gamma_{D}, \Gamma_{0}, \Delta_{0}, \Gamma_{2}, \Delta_{2}`

speed-dependent Nelkin-Ghatak Profile (SDNGP):  :math:`\Gamma_{D}, \Gamma_{0}, \Delta_{0}, \nu_{VC}, \Gamma_{2}, \Delta_{2}`

Hartmann-Tran (HTP): :math:`\Gamma_{D}, \Gamma_{0}, \Delta_{0}, \nu_{VC}, \Gamma_{2}, \Delta_{2}, \eta`

Line Intensity
--------------
The line intensity for each line at the experimental temperature is calculated using the EnvironmentDependency_Intensity function in HAPI.  This function takes as arguments the line intensity at 296 K (:math:`S(T_{ref})`), the experimental temperature (:math:`T`), the reference temperature 296 K (:math:`T_{ref}`), the partition function at the experimental temperature (:math:`Q(T)`), the partition function at the reference temperature (:math:`Q(T_{ref})`), the lower state energy (:math:`E"`), and the line center ((:math:`\nu`)), and constant (:math:`c2 = hc/k` = 1.4388028496642257).  The partition functions are calculated using `TIPS-2017 <http://dx.doi.org/10.1016/j.jqsrt.2017.03.045>`_.

.. math::

    S(T) = S(T_{ref}) \frac{Q(T_{ref})}{Q(T)}\frac{e^{-c2E"/T}}{e^{-c2E"/T_{ref}}} \frac{1 - e^{-c2\nu / T}}{1 - e^{-c2\nu / T_{ref}}}

.. todo::

    The MATS line intensity calculation currently uses the HAPI EnvironmentDependency_Intensity function, which has specific values for physical constants. The second ratiation constant, c2, is defined as hc/k and hardwired into the code.  This value includes the Boltzmann constant, which was recently redefined by the CIPM, for the sake of maintining consistency with the SI the next version of MATS will update to use the CODATA values and move from the use of the HAPI EnvironmentDependency_Intensity function.

	`CODATA: Boltzmann Constant <https://physics.nist.gov/cgi-bin/cuu/Value?k>`_



Doppler Broadening
------------------
In MATS, the doppler broadening (:math:`\Gamma_{D}`)is not a floatable parameter and is calculated based on the experimental temperature (:math:`T`), line center (:math:`\nu`), and molecular mass (:math:`m`).  The doppler width is calculated as:

.. math::

	\Gamma_{D} = \sqrt{\frac{2kT \cdot ln(2)}{cMassMol\cdot mc^{2}}} \cdot \nu

	k = 1.380648813 x 10^{-16} erg K^{-1}

	cMassMol = 1.66053873x 10^{-24} mol
.. todo::

    MATS uses the above values for the Boltzmann constant and the term cMassMol, which is equal to the inverse of Avogadro's constant.  This is consistent with the HAPI calculation of the doppler width.  However, the Boltzmann and Avogadro's constants were recently redefined by the CIPM, for the sake of maintining consistency with the SI the next version of MATS will update to use the CODATA values.  This will involve re-defining the doppler width to include Avogadro's number and not the cMassMol value.

	`CODATA: Boltzmann Constant <https://physics.nist.gov/cgi-bin/cuu/Value?k>`_

	`CODATA: Avogadro constant <https://physics.nist.gov/cgi-bin/cuu/Value?na>`_


Collisional Half-Width
----------------------
The collisional half-width (:math:`\Gamma_{0}`) is a function of both the experimental pressure (:math:`P`) and temperature (:math:`T`) referenced to :math:`P_{ref}` (1 atm) and :math:`T_{ref}` (296 K). The contributions from each diluent (k) can be scaled by the diluent composition fraction (:math:`abun`) and summed to model the ensemble collisional broadening.  The temperature dependence is modeled as a power law, where :math:`n` is the temperature exponent for the collisional width.  The collisional half-width for each line at experimental temperature and pressure can be represented as:

.. math::

	\Gamma_{0} (P,T) = \sum abun_{k} (\Gamma_{0}^{k} * \frac{P}{P_{ref}} * (\frac{T_{ref}}{T})^{n_{\Gamma_{0}^{k}}})


In the MATS nomenclature, the collisional half-width is called gamma0_diluent and the temperature dependence of the collisional half-width is called n_gamma0_diluent.


Pressure Shift
--------------
Just like the collisional half-width, the pressure shift (:math:`\Delta_{0}`) is a function of both the experimental pressure (:math:`P`) and temperature (:math:`T`) referenced to :math:`P_{ref}` (1 atm) and :math:`T_{ref}` (296 K). The contributions from each diluent (:math:`k`) can be scaled by the diluent composition fraction (:math:`abun`) and summed to model the ensemble pressure shift.  Unlike the collisional half-width, the pressure shift has a linear temperature dependence, where :math:`n` represents the temperature dependence of the pressure shift.  The pressure shift for each line at experimental pressure and temperature can be represented as:

.. math::

	\Delta_{0} (P,T) = \sum abun_{k} (\Delta_{0}^{k} +  n_{\Delta_{0}^{k}}\cdot (T - T_{ref}) )\frac{P}{P_{ref}}


In the MATS nomenclature, the pressure shift is called delta0_diluent and the temperature dependence of the pressure shift is called n_delta0_diluent.


Speed-Dependent Broadening
--------------------------
The speed-dependent mechanism accounts for the speed-dependence of relaxation rates and is parameterized in the speed-dependent Voigt (SDVP), speed-dependent Nelkin-Ghatak (SDNGP), and Hartmann-Tran (HTP) profiles.  The speed-dependent broadening in MATS is tabulated as the ratio :math:`a{w} = \frac{\Gamma_{2}}{\Gamma_{0}}`, but the actual fitted parameter is :math:`\Gamma_{2}`.  The temperature dependence of the speed-dependent broadening is a power law dependence on :math:`\Gamma_{2}`.  Currently in HITRAN, it is assumed that the :math:`n_{\Gamma_{0}} = n_{\Gamma_{2}}`, such that :math:`a_{w}` is assumed to be temperature independent.  The introduction of :math:`n_{\Gamma_{2}}` as a parameter in MATS allows for the option of this assumption to imposed, but the flexibility to explore non-equivalent temperature dependences between the speed-dependent and collisional broadening terms.  The contributions from each diluent (:math:`k`) can be scaled by the diluent composition fraction (:math:`abun`) and summed to model the ensemble speed-dependent broadening.


.. math::

	\Gamma_{2} (P,T) = \sum abun_{k} (a_{w}^{k} *\Gamma_{0}^{k} * \frac{P}{P_{ref}} * (\frac{T_{ref}}{T})^{n_{\Gamma_{2}^{k}}})


In the MATS nomenclature, the ratio of the speed-dependent broadening to the collisional broadening (:math:`a_{w}`) is called SD_gamma_diluent and the temperature dependence of the speed-dependent broadening is called n_gamma2_diluent.  The difference in the naming structure (SD_gamma vs gamma2) is chosen to emphasize the difference between the speed-dependent width being parameterized as a ratio versus as an absolute value.

Speed-Dependent Shifting
------------------------
The speed-dependent mechanism accounts for the speed-dependence of relaxation rates and is parameterized in the speed-dependent Voigt (SDVP), speed-dependent Nelkin-Ghatak (SDNGP), and Hartmann-Tran (HTP) profiles.  The speed-dependent shift in MATS is tabulated as the ratio :math:`a{s} = \frac{\Delta_{2}}{\Delta_{0}}`, but the actual fitted parameter is :math:`\Delta_{2}`.  The temperature dependence of the speed-dependent shift is modeled with a linear dependence.  Currently, the temperature dependence of the speed-dependent shift is not parameterized in HITRAN.  The contributions from each diluent (:math:`k`) can be scaled by the diluent composition fraction (:math:`abun`) and summed to model the ensemble speed-dependent shift.


.. math::

	\Delta_{2} (P,T) = \sum abun_{k} (a_{s} \cdot \Delta_{0}^{k} +  n_{\Delta_{2}^{k}}\cdot (T - T_{ref}) )\frac{P}{P_{ref}}

In MATS nomenclature, the ratio of the speed-dependent shift to the pressure shift (:math:`a_{s}`) is called SD_shift_diluent and the temperature dependence of the speed-dependent shift is called n_delta2_diluent.  The difference in the naming structure (SD_delta vs delta2) is chosen to emphasize the difference between the speed-dependent shift being parameterized as a ratio versus as an absolute value.


Dicke Narrowing
---------------
The Dicke narrowing mechanism models collisional induced velocity changes and is parameterized in the Nelkin-Ghatak (NGP), speed-dependent Nelkin-Ghatak (SDNGP), and Hartmann-Tran (HTP) profiles by the term :math:`\nu_{VC}`. The temperature dependence is modeled as a power law, where :math:`n` represents the temperature dependence of the Dicke narrowing term.  If the Dicke narrowing is assumed to behave like the diffusion coefficient, then the temperature dependence theoretically should be 1.  The contributions from each diluent (:math:`k`) can be scaled by the diluent composition fraction (:math:`abun`) and summed to model the ensemble Dicke narrowing.

.. math::

	\nu_{VC} (P,T) = \sum_{k=i} abun_{k} (\nu_{VC}^{k} * \frac{P}{P_{ref}} * (\frac{T_{ref}}{T})^{n_{\nu{VC}^{k}}})


In MATS nomenclature, the Dicke narrowing is referred to as nuVC_diluent and the temperature exponent is n_nuVC_diluent.  This differs from the naming convention in HAPI, which changes based on the origin of the Dicke narrowing term (Galatry profile versus HTP).  For simplicity, MATS has adopted a self-consistent naming convention.


Correlation Parameter
---------------------
The correlation parameter (:math:`\eta`) models the correlation between velocity and rotation state changes due to collisions and is only parameterized in the Hartmann-Tran profile (HTP).  Currently, MATS has no temperature or pressure dependence associated with the correlation parameter.  However, contributions from each diluent (:math:`k`) can be scaled by the diluent composition fraction (:math:`abun`) and summed to model the ensemble correlation parameter.

.. math::

	\eta (k) = \sum abun_{k} \cdot \eta

In MATS nomenclature, the correlation parameter is referred to as eta_diluent.

Line Mixing
-----------
The nearest-neighbor line mixing (:math:`Y`) can be calculated from the imaginary portion of any of the HTP derivative line profiles.  Currently, there is no temperature dependence imposed on the line mixing, so there a different value is used for each nominal temperature, where the nominal temperature is specified in the :py:class:`MATS.Spectrum` definition.  However, contributions from each diluent (:math:`k`) can be scaled by the diluent composition fraction (:math:`abun`) and  summed to model the ensemble line mixing.

.. math::

	Y (P) = \sum abun_{k} Y \frac{P}{P_{ref}}

The line mixing is implemented as:

.. math::

	\alpha = I * (Re{HTP(\Gamma_{D}, \Gamma_{0}, \Delta_{0}, \Gamma_{2}, \Delta_{2}, \nu_{VC}, \eta, \nu)} + Y*Im{HTP(\Gamma_{D}, \Gamma_{0}, \Delta_{0}, \Gamma_{2}, \Delta_{2}, \nu_{VC}, \eta, \nu)})

In MATS nomenclature, the line mixing parameter is referred to as y_diluent_nominaltemperature.


MATS Functions
--------------

.. currentmodule:: MATS.HTP_from_DF_select

.. autosummary::
   MATS.HTP_from_DF_select
   MATS.etalon



Dataset Class
++++++++++++++

.. currentmodule:: MATS.Dataset

.. autosummary::
   MATS.Dataset
   MATS.Dataset.generate_baseline_paramlist
   MATS.Dataset.generate_summary_file
   MATS.Dataset.get_spectra_extremes
   MATS.Dataset.get_spectrum_extremes
   MATS.Dataset.average_QF
   MATS.Dataset.plot_model_residuals

Generate FitParam File Class
++++++++++++++++++++++++++++
.. currentmodule:: MATS.Generate_FitParam_File

.. autosummary::
   MATS.Generate_FitParam_File
   MATS.Generate_FitParam_File.generate_fit_baseline_linelist
   MATS.Generate_FitParam_File.generate_fit_param_linelist_from_linelist



Edit Fit Param Files Class
++++++++++++++++++++++++++
.. currentmodule:: MATS.Edit_Fit_Param_Files

.. autosummary::
   MATS.Edit_Fit_Param_Files
   MATS.Edit_Fit_Param_Files.edit_generated_baselist
   MATS.Edit_Fit_Param_Files.edit_generated_paramlist
   MATS.Edit_Fit_Param_Files.save_editted_baselist
   MATS.Edit_Fit_Param_Files.save_editted_paramlist


Fit Dataset Class
+++++++++++++++++

.. currentmodule:: MATS.Fit_DataSet

.. autosummary::
   MATS.Fit_DataSet
   MATS.Fit_DataSet.constrained_baseline
   MATS.Fit_DataSet.fit_data
   MATS.Fit_DataSet.generate_params
   MATS.Fit_DataSet.residual_analysis
   MATS.Fit_DataSet.simulation_model
   MATS.Fit_DataSet.update_params
