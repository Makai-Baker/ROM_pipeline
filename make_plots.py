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
import seaborn as sns
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
label = inputs.label

if ini_args.frequency_domain_source_model == "lal_binary_black_hole":
    cbc_type = 'bbh'
else:
    cbc_type = 'bns'

degrees = ['linear', 'quadratic']

#model = RombusModel.load('{}_model_linear:GWModel'.format(cbc_type))

time_dependent=True
duration = float(ini_args.duration)
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

f_min, f_max = float(ini_args.minimum_frequency), float(ini_args.maximum_frequency)
uniform_f = np.linspace(f_min, f_max, int((f_max - f_min)*duration) + 1 )
psd = ifo.power_spectral_density.power_spectral_density_interpolated(uniform_f)


for i in range(len(degrees)):
    degree = degrees[i]
    print(degree)
    #samples._add_from_file('samples.npy')
    set_params=True
    if i>0:
        set_params=False

    model = GWModel(ini_args, inputs, cbc_type, degree, ifo, set_params)#RombusModel.load('model:GWModel',ini_args, inputs, cbc_type, degree, ifo)#

    samples = Samples(model=model, n_random=1)
    training_samples = utils.random_samples(model, n_random=10000) #out of TS samples.

    training_samples = training_samples[:len(training_samples)]
    samples.extend(training_samples)
    samples.samples = samples.samples[1:]
    samples.n_samples -= 1
    
    mismatch = utils.calculate_mismatch(
        model, 
        samples.samples, 
        '{}_B_{}.npy'.format(cbc_type, degree), #'interpolated_B_linear.npy',#
        '{}_fnodes_{}.npy'.format(cbc_type, degree),
        psd)

    mismatches = rombus._core.mpi.COMM.gather(mismatch, root=rombus._core.mpi.MAIN_RANK)

    if rombus._core.mpi.RANK_IS_MAIN:
        all_mismatch = [item for sublist in mismatches for item in sublist]
        sns.histplot(all_mismatch, bins=int(np.sqrt(len(all_mismatch))), log_scale=True)
        plt.gca().set(xlabel='Mismatch')
        plt.savefig('{}_mismatch_{}.pdf'.format(cbc_type,degree))
        plt.clf()

        lin_greedy = np.load('greedy_params.npy')
        utils.plot_greedypoints(model, lin_greedy, 'chirp_mass', 'chi1L') 
        
        #training_samples = np.load('samples.npy', allow_pickle=True)
        #training_samples = training_samples[:len(training_samples)//10]
        #training_samples = np.array(training_samples)
        #mismatch = np.array(all_mismatch)
        #mask = np.where(all_mismatch > 1e-2)
        #utils.plot_greedypoints(model, training_samples[mask], 'chirp_mass', 'chi1L', all_mismatch[mask], label='mismatch_points')