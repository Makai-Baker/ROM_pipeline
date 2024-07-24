import rombus
from rombus.model import RombusModel
from rombus.samples import Samples
from rombus.rom import ReducedOrderModel
import subprocess
import numpy as np
import argparse
from bilby_pipe.parser import create_parser
from bilby_pipe.main import parse_args, MainInput
import utils
import seaborn as sns
import matplotlib.pyplot as plt

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

degree = 'linear'

#model = RombusModel.load('{}_model_linear:GWModel'.format(cbc_type))
model = RombusModel.load('model:GWModel'.format(cbc_type))

samples = Samples(model=model, n_random=1)
#training_samples = np.load('samples.npy', allow_pickle=True)

training_samples = utils.random_samples(model, n_random=10000) #out of TS samples.

#new_ts = set(tuple(item) for item in new_samples)
#ts = set(tuple(item) for item in training_samples)

#out_of_ts = [list(item) for item in new_ts.difference(ts)]
#print(len(out_of_ts))
#print(out_of_ts[0])

training_samples = training_samples[:len(training_samples)]
samples.extend(training_samples)
samples.samples = samples.samples[1:]
samples.n_samples -= 1

#samples._add_from_file('samples.npy')
mismatch = utils.calculate_mismatch(
    model, 
    samples.samples, 
    '{}_B_linear.npy'.format(cbc_type), #'interpolated_B_linear.npy',#
    '{}_fnodes_linear.npy'.format(cbc_type))

mismatches = rombus._core.mpi.COMM.gather(mismatch, root=rombus._core.mpi.MAIN_RANK)

if rombus._core.mpi.RANK_IS_MAIN:
    all_mismatch = [item for sublist in mismatches for item in sublist]
    sns.histplot(all_mismatch, bins=int(np.sqrt(len(all_mismatch))), log_scale=True)
    plt.gca().set(xlabel='Mismatch')
    plt.savefig('{}_mismatch.pdf'.format(cbc_type))
    plt.clf()

    lin_greedy = np.load('greedy_params.npy')
    utils.plot_greedypoints(model, lin_greedy, 'chirp_mass', 'chi1L') 
    
    #training_samples = np.load('samples.npy', allow_pickle=True)
    #training_samples = training_samples[:len(training_samples)//10]
    #training_samples = np.array(training_samples)
    #mismatch = np.array(all_mismatch)
    #mask = np.where(all_mismatch > 1e-2)
    #utils.plot_greedypoints(model, training_samples[mask], 'chirp_mass', 'chi1L', all_mismatch[mask], label='mismatch_points')