# from MATS.hapi import *
# from MATS.MATS import *

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from MATS import MATS
from MATS.exampledata import exampledata


class FixtureBase(object):
    def __init__(self, seed) -> None:
        self.seed = seed
        np.random.seed(seed)
        self._init_sub()

    def _init_sub(self):
        raise NotImplementedError

    @staticmethod
    def params_to_frame(params):
        df = pd.DataFrame(
            [
                dict(
                    name=v.name,
                    value=v.value,
                    init_value=v.init_value,
                    min=v.min,
                    max=v.max,
                    vary=v.vary,
                    stderr=v.stderr,
                )
                for k, v in params.items()
            ]
        ).set_index("name")
        return df


class FixtureSimulation(FixtureBase):
    def _init_sub(self):
        PARAM_LINELIST = exampledata.O2_ABand_Drouin_2017_linelist

        # Generic Fit Parameters
        wave_range = 1.5  # range outside of experimental x-range to simulate
        IntensityThreshold = (
            1e-30  # intensities must be above this value to be simulated
        )
        Fit_Intensity = (
            1e-24  # intensities must be above this value for the line to be fit
        )
        order_baseline_fit = 1
        sample_molefraction = {7: 0.002022}
        wave_min = 13155  # cm-1
        wave_max = 13157.5  # cm-1
        wave_space = 0.005  # cm-1
        baseline_terms = [
            0
        ]  # polynomial baseline coefficients where the index is equal to the coefficient order

        # Error Sources
        ##SNR
        SNR = 10000

        ## Wavenumber - adds gaussian noise of given absolute magniture
        wave_error = 5e-5

        ## Temperature
        temperature_err = {"bias": 0.01, "function": None, "params": {}}

        ## Pressure
        pressure_err = {"per_bias": 0.01, "function": None, "params": {}}
        ##Mole Fraction
        molefraction_err = {7: 0.01}

        def get_simulate_spectrum(pressure, filename):
            return MATS.simulate_spectrum(
                PARAM_LINELIST,
                wave_min,
                wave_max,
                wave_space,
                wave_error=wave_error,
                SNR=SNR,
                baseline_terms=baseline_terms,
                temperature=25,
                temperature_err=temperature_err,
                pressure=pressure,
                pressure_err=pressure_err,
                wing_cutoff=50,
                wing_method="wing_cutoff",
                filename=filename,
                molefraction=sample_molefraction,
                molefraction_err=molefraction_err,
                natural_abundance=True,
                nominal_temperature=296,
                IntensityThreshold=1e-30,
                num_segments=1,
            )

        spec_1 = get_simulate_spectrum(pressure=20, filename="20_torr")

        spec_2 = get_simulate_spectrum(
            pressure=40,
            filename="40_torr",
        )
        spec_3 = get_simulate_spectrum(
            pressure=60,
            filename="60_torr",
        )
        spec_4 = get_simulate_spectrum(
            pressure=80,
            filename="80_torr",
        )

        # Add all spectrum to a Dataset object
        SPECTRA = MATS.Dataset(
            [spec_1, spec_2, spec_3, spec_4],
            "Line Intensity",
            baseline_order=order_baseline_fit,
        )

        # Generate Baseline Parameter list based on number of etalons in spectra definitions and baseline order
        BASE_LINELIST = SPECTRA.generate_baseline_paramlist()

        FITPARAMS = MATS.Generate_FitParam_File(
            SPECTRA,
            PARAM_LINELIST,
            BASE_LINELIST,
            lineprofile="SDVP",
            linemixing=False,
            fit_intensity=Fit_Intensity,
            threshold_intensity=IntensityThreshold,
            sim_window=wave_range,
            nu_constrain=False,
            sw_constrain=False,
            gamma0_constrain=True,
            delta0_constrain=True,
            aw_constrain=True,
            as_constrain=True,
            nuVC_constrain=True,
            eta_constrain=True,
            linemixing_constrain=True,
        )

        FITPARAMS.generate_fit_param_linelist_from_linelist(
            vary_nu={7: {1: True, 2: False, 3: False}},
            vary_sw={7: {1: True, 2: False, 3: False}},
            vary_gamma0={7: {1: True, 2: False, 3: False}, 1: {1: False}},
            vary_n_gamma0={7: {1: True}},
            vary_delta0={7: {1: False, 2: False, 3: False}, 1: {1: False}},
            vary_n_delta0={7: {1: True}},
            vary_aw={7: {1: True, 2: False, 3: False}, 1: {1: False}},
            vary_n_gamma2={7: {1: False}},
            vary_as={},
            vary_n_delta2={7: {1: False}},
            vary_nuVC={7: {1: False}},
            vary_n_nuVC={7: {1: False}},
            vary_eta={},
            vary_linemixing={7: {1: False}},
        )

        FITPARAMS.generate_fit_baseline_linelist(
            vary_baseline=True,
            vary_molefraction={7: False, 1: False},
            vary_xshift=False,
            vary_etalon_amp=True,
            vary_etalon_freq=False,
            vary_etalon_phase=True,
            vary_pressure=False,
            vary_temperature=False,
        )

        fit_data = MATS.Fit_DataSet(
            SPECTRA,
            "Baseline_LineList",
            "Parameter_LineList",
            minimum_parameter_fit_intensity=Fit_Intensity,
            baseline_limit=False,
            baseline_limit_factor=10,
            molefraction_limit=False,
            molefraction_limit_factor=1.1,
            etalon_limit=False,
            etalon_limit_factor=2,  # phase is constrained to +/- 2pi,
            x_shift_limit=False,
            x_shift_limit_magnitude=0.5,
            nu_limit=False,
            nu_limit_magnitude=0.1,
            sw_limit=False,
            sw_limit_factor=2,
            gamma0_limit=False,
            gamma0_limit_factor=3,
            n_gamma0_limit=False,
            n_gamma0_limit_factor=50,
            delta0_limit=False,
            delta0_limit_factor=2,
            n_delta0_limit=False,
            n_delta0_limit_factor=50,
            SD_gamma_limit=False,
            SD_gamma_limit_factor=2,
            n_gamma2_limit=False,
            n_gamma2_limit_factor=50,
            SD_delta_limit=False,
            SD_delta_limit_factor=50,
            n_delta2_limit=False,
            n_delta2_limit_factor=50,
            nuVC_limit=False,
            nuVC_limit_factor=2,
            n_nuVC_limit=False,
            n_nuVC_limit_factor=50,
            eta_limit=False,
            eta_limit_factor=50,
            linemixing_limit=False,
            linemixing_limit_factor=50,
        )
        params = fit_data.generate_params()

        result = fit_data.fit_data(params, wing_cutoff=25, wing_wavenumbers=1)

        self.PARAM_LINELIST = PARAM_LINELIST

        self.SPECTRA = SPECTRA

        self.BASE_LINELIST = BASE_LINELIST

        self.FITPARAMS = FITPARAMS

        self.fit_data = fit_data

        self.params = params
        self.result = result


@pytest.fixture(params=[(0,)], scope="module")
def sim_fix(request):
    return FixtureSimulation(*request.param)


class FixtureData(FixtureBase):
    def _init_sub(self):

        # Generic Fit Parameters
        wave_range = 1.5  # range outside of experimental x-range to simulate
        IntensityThreshold = (
            1e-30  # intensities must be above this value to be simulated
        )
        Fit_Intensity = (
            1e-24  # intensities must be above this value for the line to be fit
        )
        order_baseline_fit = 1
        tau_column = "Corrected Tau (us)"  # Mean tau/us
        freq_column = "Total Frequency (Detuning)"  # Total Frequency /MHz
        pressure_column = "Cavity Pressure /Torr"
        temperature_column = "Cavity Temperature Side 2 /C"

        def get_spectrum(path):
            # Define all Spectra individually
            return MATS.Spectrum(
                path,
                molefraction={7: 0.01949},
                natural_abundance=True,
                diluent="air",
                etalons={1: [0.001364, 1.271443]},
                input_freq=True,
                frequency_column=freq_column,
                input_tau=True,
                tau_column=tau_column,
                tau_stats_column=None,
                pressure_column=pressure_column,
                temperature_column=temperature_column,
                nominal_temperature=296,
                x_shift=0.00,
            )

        root = Path(__file__).parent

        paths = [
            str(root / name)
            for name in [
                "190510_2per_43_forfit",
                "190510_2per_55_forfit",
                "190513_2per_82_forfit",
                "190514_2per_126_forfit",
            ]
        ]

        self.paths = paths

        # Define all Spectra individually
        spec_1 = get_spectrum(paths[0])
        spec_2 = get_spectrum(paths[1])
        spec_3 = get_spectrum(paths[2])
        spec_4 = get_spectrum(paths[3])

        # Add all spectrum to a Dataset object
        SPECTRA = MATS.Dataset(
            [spec_1, spec_2, spec_3, spec_4],
            "Line Intensity",
            baseline_order=order_baseline_fit,
        )

        # Generate Baseline Parameter list based on number of etalons in spectra definitions and baseline order
        BASE_LINELIST = SPECTRA.generate_baseline_paramlist()
        # BASE_LINELIST = pd.read_csv('Line Intensity_baseline_paramlist.csv')

        # Read in Possible linelists
        # hapi = r'C:\Users\ema3\Documents\MATS\MATS\Linelists'
        # os.chdir(hapi)
        # PARAM_LINELIST = pd.read_csv('O2_ABand_Drouin_2017_linelist.csv')
        PARAM_LINELIST = exampledata.O2_ABand_Drouin_2017_linelist

        # os.chdir(path)
        # Set-up for Fitting
        # lineprofile = 'NGP' #VP, SDVP, NGP, SDNGP, HTP

        FITPARAMS = MATS.Generate_FitParam_File(
            SPECTRA,
            PARAM_LINELIST,
            BASE_LINELIST,
            lineprofile="SDVP",
            linemixing=False,
            fit_intensity=Fit_Intensity,
            threshold_intensity=IntensityThreshold,
            sim_window=wave_range,
            nu_constrain=False,
            sw_constrain=False,
            gamma0_constrain=True,
            delta0_constrain=True,
            aw_constrain=True,
            as_constrain=True,
            nuVC_constrain=True,
            eta_constrain=True,
            linemixing_constrain=True,
        )

        FITPARAMS.generate_fit_param_linelist_from_linelist(
            vary_nu={7: {1: True, 2: False, 3: False}},
            vary_sw={7: {1: True, 2: False, 3: False}},
            vary_gamma0={7: {1: True, 2: False, 3: False}, 1: {1: False}},
            vary_n_gamma0={7: {1: True}},
            vary_delta0={7: {1: False, 2: False, 3: False}, 1: {1: False}},
            vary_n_delta0={7: {1: True}},
            vary_aw={7: {1: True, 2: False, 3: False}, 1: {1: False}},
            vary_n_gamma2={7: {1: False}},
            vary_as={},
            vary_n_delta2={7: {1: False}},
            vary_nuVC={7: {1: False}},
            vary_n_nuVC={7: {1: False}},
            vary_eta={},
            vary_linemixing={7: {1: False}},
        )

        FITPARAMS.generate_fit_baseline_linelist(
            vary_baseline=True,
            vary_molefraction={7: False, 1: False},
            vary_xshift=False,
            vary_etalon_amp=True,
            vary_etalon_freq=False,
            vary_etalon_phase=True,
        )

        # os.chdir(path)
        fit_data = MATS.Fit_DataSet(
            SPECTRA,
            "Baseline_LineList",
            "Parameter_LineList",
            minimum_parameter_fit_intensity=Fit_Intensity,
            baseline_limit=False,
            baseline_limit_factor=10,
            molefraction_limit=False,
            molefraction_limit_factor=1.1,
            etalon_limit=False,
            etalon_limit_factor=2,  # phase is constrained to +/- 2pi,
            x_shift_limit=False,
            x_shift_limit_magnitude=0.5,
            nu_limit=True,
            nu_limit_magnitude=0.1,
            sw_limit=True,
            sw_limit_factor=2,
            gamma0_limit=False,
            gamma0_limit_factor=3,
            n_gamma0_limit=False,
            n_gamma0_limit_factor=50,
            delta0_limit=False,
            delta0_limit_factor=2,
            n_delta0_limit=False,
            n_delta0_limit_factor=50,
            SD_gamma_limit=False,
            SD_gamma_limit_factor=2,
            n_gamma2_limit=False,
            n_gamma2_limit_factor=50,
            SD_delta_limit=False,
            SD_delta_limit_factor=50,
            n_delta2_limit=False,
            n_delta2_limit_factor=50,
            nuVC_limit=False,
            nuVC_limit_factor=2,
            n_nuVC_limit=False,
            n_nuVC_limit_factor=50,
            eta_limit=False,
            eta_limit_factor=50,
            linemixing_limit=False,
            linemixing_limit_factor=50,
        )
        params = fit_data.generate_params()

        for param in params:
            if "SD_gamma" in param:
                params[param].set(min=0.01, max=0.25)
            if "etalon_1_amp" in param:
                if param != "etalon_1_amp_1_1":
                    params[param].set(expr="etalon_1_amp_1_1")

        result = fit_data.fit_data(params, wing_cutoff=25)

        self.PARAM_LINELIST = PARAM_LINELIST

        self.SPECTRA = SPECTRA

        self.BASE_LINELIST = BASE_LINELIST

        self.FITPARAMS = FITPARAMS

        self.fit_data = fit_data

        self.params = params
        self.result = result


@pytest.fixture(params=[(0,)], scope="module")
def exp_fix(request):
    return FixtureData(*request.param)
