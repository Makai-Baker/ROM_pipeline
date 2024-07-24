from typing import NamedTuple

import lal  # type: ignore
import lalsimulation  # type: ignore
import numpy as np
import bilby 
from rombus.model import RombusModel
import matplotlib.pyplot as plt
from __main__ import ini_args, args, inputs, cbc_type, degree

from utils import banded_sampling, get_component_masses_range

#from imports_for_model import param_ranges, parameters, f_min, f_max

class GWModel(RombusModel):

    f_min = float(ini_args.minimum_frequency)
    f_max = float(ini_args.maximum_frequency)

    delta_F = 1/float(ini_args.duration)
    n_f = int((f_max - f_min) / delta_F) + 1

    try:
        priors = bilby.core.prior.PriorDict(ini_args.prior_file)
    except Exception as e:
        priors = inputs.prior_dict

    ordinate.set('h', label='$h$', dtype=complex)

    coordinate.set('f', min=f_min, max=f_max, n_values=n_f, label="$f$", dtype=np.dtype("float64"))

    ref_freq = float(ini_args.reference_frequency)

    params.add('chirp_mass', priors['chirp_mass'].minimum, priors['chirp_mass'].maximum)
    params.add('mass_ratio', priors['mass_ratio'].minimum, priors['mass_ratio'].maximum)
    params.add("chi1L", priors['chi_1'].minimum, priors['chi_1'].maximum)
    params.add("chi2L", priors['chi_2'].minimum, priors['chi_2'].maximum)
    params.add("chip", priors['chi_p'].minimum, priors['chi_p'].maximum)  
    params.add("ld", priors['luminosity_distance'].minimum, priors['luminosity_distance'].maximum)
    params.add("thetaJ", 0, np.pi)  
    params.add("alpha", 0, 2*np.pi)
    params.add("phase", 0, 2*np.pi)

    params.add('ra', 0, 2*np.pi)
    params.add('dec', -np.pi/2, np.pi/2)
    params.add('geocent_time', prior['geocent_time'].minimum, prior['geocent_time'].maximum)
    parans.add('psi', 0, np.pi)

    if cbc_type == 'bns':
        params.add("l1", priors['lambda_1'].minimum, priors['lambda_1'].maximum)
        params.add("l2", priors['lambda_2'].minimum, priors['lambda_2'].maximum) 

    def compute(self, params, domain: np.ndarray) -> np.ndarray:
        WFdict = lal.CreateDict()

        chirp_mass = params.chirp_mass
        mass_ratio = params.mass_ratio

        m1, m2 = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(chirp_mass, mass_ratio)

        if cbc_type == 'bns':
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
            1e6 * lal.lal.PC_SI * params.ld,
            params.alpha, 
            params.phase,
            self.ref_freq, #reference frequency
            approx,
            tides,
            WFdict,
        )

        ## Build antenna response.
        fp = bilby.gw.detector.interferometer.antenna_response(params.ra, params.dec, params.geocent_time, params.psi, 'plus')
        fc = bilby.gw.detector.interferometer.antenna_response(params.ra, params.dec, params.geocent_time, params.psi, 'cross')

        hp, hc = h[0].data.data, h[1].data.data
        h = fp*hp + fc*hc 
        # possibly add in phase term e^{2 \pi i f dt}
        if degree == 'linear':
            return h #hp+hc
        else:
            
            #hp_hc = hp+hc 
            return np.conjugate(h)*h #np.conjugate(hp_hc)*hp_hc
            