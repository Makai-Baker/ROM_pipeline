from typing import NamedTuple

import lal  # type: ignore
import lalsimulation  # type: ignore
import numpy as np
import bilby 
from rombus.model import RombusModel
import matplotlib.pyplot as plt
#from __main__ import ini_args, inputs, cbc_type, degree, ifo

from utils import banded_sampling, get_component_masses_range

#from imports_for_model import param_ranges, parameters, f_min, f_max

class GWModel(RombusModel):

    def __init__(self, ini_args, inputs, cbc_type, degree, ifo, set_params):
        self.ini_args = ini_args 
        self.inputs = inputs 
        self.cbc_type = cbc_type 
        self.degree = degree 
        self.ifo = ifo

        self.f_min = float(ini_args.minimum_frequency)
        self.f_max = float(ini_args.maximum_frequency)
        self.duration = float(ini_args.duration)
        self.sampling_frequency = float(ini_args.sampling_frequency)
        self.trigger_time = float(ini_args.trigger_time)

        self.delta_F = 1/self.duration
        self.n_f = int((self.f_max - self.f_min) / self.delta_F) + 1

        self.ordinate.set('h', label='$h$', dtype=complex)

        self.coordinate.set('f', min=self.f_min, max=self.f_max, n_values=self.n_f, label="$f$", dtype=np.dtype("float64"))

        try:
            self.priors = bilby.core.prior.PriorDict(self.ini_args.prior_file)
        except Exception as e:
            self.priors = self.inputs.prior_dict

        self.ordinate.set('h', label='$h$', dtype=complex)

        self.coordinate.set('f', min=self.f_min, max=self.f_max, n_values=self.n_f, label="$f$", dtype=np.dtype("float64"))

        self.ref_freq = float(self.ini_args.reference_frequency)

        self.ld = 100

        if set_params:
            self.set_params()
    # params.add('chirp_mass', 1,2)
    # params.add('mass_ratio', 1,2)
    # params.add("chi1L", 0,1)
    # params.add("chi2L", 0,1)
    # params.add("chip", 0,1)  
    # #params.add("ld", priors['luminosity_distance'].minimum, priors['luminosity_distance'].maximum)
    # ld = 100.
    # params.add("thetaJ", 0, np.pi)  
    # params.add("alpha", 0, 2*np.pi)
    # params.add("phase", 0, 2*np.pi)

    # params.add('ra', 0, 2*np.pi)
    # params.add('dec', -np.pi/2, np.pi/2)
    # params.add('geocent_time', 0,0)
    # params.add('psi', 0, np.pi)

    # params.add("l1", 0,1)
    # params.add("l2", 0,1) 
    def set_params(self):
        self.params.add('chirp_mass', self.priors['chirp_mass'].minimum, self.priors['chirp_mass'].maximum)
        self.params.add('mass_ratio', self.priors['mass_ratio'].minimum, self.priors['mass_ratio'].maximum)
        self.params.add("chi1L", self.priors['chi_1'].minimum, self.priors['chi_1'].maximum)
        self.params.add("chi2L", self.priors['chi_2'].minimum, self.priors['chi_2'].maximum)
        self.params.add("chip", self.priors['chi_p'].minimum, self.priors['chi_p'].maximum)  
        #params.add("ld", priors['luminosity_distance'].minimum, priors['luminosity_distance'].maximum)
        self.params.add("thetaJ", 0, np.pi)  
        self.params.add("alpha", 0, 2*np.pi)
        self.params.add("phase", 0, 2*np.pi)

        self.params.add('ra', 0, 2*np.pi)
        self.params.add('dec', -np.pi/2, np.pi/2)
        self.params.add('geocent_time', self.priors['geocent_time'].minimum, self.priors['geocent_time'].maximum)
        self.params.add('psi', 0, np.pi)

        if self.cbc_type == 'bns':
            self.params.add("l1", self.priors['lambda_1'].minimum, self.priors['lambda_1'].maximum)
            self.params.add("l2", self.priors['lambda_2'].minimum, self.priors['lambda_2'].maximum) 
        
    def compute(self, params, domain: np.ndarray) -> np.ndarray:
        WFdict = lal.CreateDict()

        chirp_mass = params.chirp_mass
        mass_ratio = params.mass_ratio

        m1, m2 = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(chirp_mass, mass_ratio)

        if self.cbc_type == 'bns':
            lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(WFdict, params.l1)
            lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(WFdict, params.l2)

            approx = lalsimulation.IMRPhenomPv2NRTidal_V
            tides = lalsimulation.NRTidalv2_V 
        
        else: 
            approx = lalsimulation.IMRPhenomPv2_V
            tides = lalsimulation.NoNRT_V
       
        h = lalsimulation.SimIMRPhenomPFrequencySequence(
            domain,
            params.chi1L,  # type: ignore
            params.chi2L,  # type: ignore
            params.chip,  # type: ignore
            params.thetaJ,  # type: ignore
            m1 * lal.lal.MSUN_SI,  # type: ignore
            m2 * lal.lal.MSUN_SI,  # type: ignore
            1e6 * lal.lal.PC_SI * self.ld,
            params.alpha, 
            params.phase,
            self.ref_freq, #reference frequency
            approx,
            tides,
            WFdict,
        )

        hp, hc = h[0].data.data, h[1].data.data
        
        wf_pols = {'plus': hp, 'cross': hc}

        ## Build antenna response.
        self.ifo.frequency_mask = bool(np.ones_like(len(domain)))
        h = self.ifo.get_detector_response(wf_pols, params._asdict(), domain)

        if self.degree == 'linear':
            return h
        else:
            print("Returning quadratic component")
            return np.conjugate(h)*h
            