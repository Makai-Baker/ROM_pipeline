import rombus
from rombus.model import RombusModel
from rombus.samples import Samples
from rombus.rom import ReducedOrderModel
import subprocess
import numpy as np
import argparse
from bilby_pipe.parser import create_parser
from bilby_pipe.main import parse_args, MainInput
import bilby
import utils
import matplotlib.pyplot as plt 
from model import GWModel 

parser = argparse.ArgumentParser()
parser.add_argument("-as", "--adaptive-sampling", help="Adaptive sampling")
parser.add_argument("-i", "--ini-file", help="Ini file")
args = parser.parse_args()

adapt_sampling = False
if args.adaptive_sampling=='True':
    adapt_sampling = True 

ini_file = args.ini_file 
parser = create_parser(top_level=True)
ini_args, unknown_args = parse_args([ini_file], parser)
inputs = MainInput(ini_args, unknown_args)

if ini_args.frequency_domain_source_model == "lal_binary_black_hole":
    cbc_type = 'bbh'
else:
    cbc_type = 'bns'

degrees = ['quadratic']

if adapt_sampling:
    degrees = ['linear', 'quadratic']

f_min, f_max, duration = float(ini_args.minimum_frequency), float(ini_args.maximum_frequency), float(ini_args.duration)

uniform_f = np.linspace(f_min, f_max, int((f_max - f_min)*duration) + 1 )

time_dependent=True
sampling_frequency = float(ini_args.sampling_frequency)
trigger_time = float(ini_args.trigger_time)
post_trigger_duration = 2.
ifos = bilby.gw.detector.InterferometerList([det for det in ini_args.detectors], time_dependent=time_dependent) #Add updated CE noise curve
#Make this work for multiple interferometers
ifo = ifos[0]
print(ifo.__class__.__name__)

start_time = trigger_time + post_trigger_duration - duration

ifo.set_strain_data_from_power_spectral_density(
    sampling_frequency=sampling_frequency, duration=duration,
    start_time=start_time)

if time_dependent:
    ifo.initialize_gmsts(trigger_time)

psd = ifo.power_spectral_density.power_spectral_density_interpolated(uniform_f)

tol = 1e-12

for i in range(len(degrees)):
    degree = degrees[i]

    set_params=True
    if i>0:
        set_params=False

    model = RombusModel.load('model:GWModel',ini_args, inputs, cbc_type, degree, ifo, set_params)#GWModel(ini_args, inputs, cbc_type, degree, ifo, set_params)

    model.domain = uniform_f 
    model.n_domain = len(uniform_f)
    up_samples = Samples(model=model, n_random=1)

    up_samples._add_from_file('greedy_params.npy')

    rom = ReducedOrderModel(model, up_samples, psd=psd).build(do_step=None, tol=tol)

    error_list = rom.reduced_basis.error_list
    plt.plot(error_list)
    plt.xlabel("# Basis elements")
    plt.ylabel("Error")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig('{}_error.pdf'.format(degree))
    print(rom.empirical_interpolant.B_matrix.shape)

    np.save('{}_B_{}'.format(cbc_type, degree), rom.empirical_interpolant.B_matrix)
    np.save('{}_fnodes_{}'.format(cbc_type, degree), rom.empirical_interpolant.nodes)