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
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-as", "--adaptive-sampling", help="Adaptive sampling")
parser.add_argument("-i", "--ini-file", help="Ini file")
parser.add_argument("-n", "--number-of-samples", help="Number of samples to populate ROM training set")

args = parser.parse_args()

adapt_sampling = False
if args.adaptive_sampling=='True':
    adapt_sampling = True 

n_random = args.number_of_samples

if n_random is None:
    n_random = 1e6  #arbitrary number. Need to look into some selection method for this.
else:
    n_random = int(n_random)

ini_file = args.ini_file 
parser = create_parser(top_level=True)
ini_args, unknown_args = parse_args([ini_file], parser)
inputs = MainInput(ini_args, unknown_args)

if ini_args.frequency_domain_source_model == "lal_binary_black_hole":
    cbc_type = 'bbh'
else:
    cbc_type = 'bns'

n_refine = 2  #See Fig 2 of https://arxiv.org/pdf/1604.08253.pdf

degree = 'linear'

lin_greedy = None 
    
label = inputs.label
#model = RombusModel.load('{}_model_{}:GWModel'.format(cbc_type,degree))
model = RombusModel.load('model:GWModel')

if adapt_sampling:
    f_values = utils.down_sample(ini_args, df_max=1.)
    n_f = model.n_domain 

    model.domain = f_values
    model.n_domain = len(f_values)

    print('Compression rate from down-sampling:', n_f/len(f_values))

samples = Samples(model=model, n_random=1)

boundary_samples = utils.boundary_samples(model)
grid_samples = utils.grid_samples(model, n_random) 
random_samples = utils.random_samples(model, n_random)

samples.extend(boundary_samples)
samples.extend(grid_samples)
samples.extend(random_samples)

tol = 1e-12

rom = ReducedOrderModel(model, samples).build(do_step=None, tol=tol)
#for _ in range(n_refine):
#    rom = rom.refine(n_random, iterate=False)

error_list = rom.reduced_basis.error_list
plt.plot(error_list)
plt.xlabel("# Basis elements")
plt.ylabel("Error")
plt.yscale("log")
plt.tight_layout()
plt.savefig('lin_ds_error.pdf')

print(rom.empirical_interpolant.B_matrix.shape)

lin_greedy = rom.reduced_basis.greedypoints
greedy_samples = [arr.tolist() for arr in lin_greedy]

if rombus._core.mpi.RANK_IS_MAIN:
    interpolate = False 
    if interpolate: 
        up_B_lin = []
        f_min = float(ini_args.minimum_frequency)
        f_max = float(ini_args.maximum_frequency) 
        duration = float(ini_args.duration)
        for i, row in enumerate(rom.empirical_interpolant.B_matrix):
            x = model.domain 
            cs = CubicSpline(x, row)
            up_domain = np.linspace(f_min, f_max, int(duration*(f_max-f_min))+1)
            up_B_lin.append(cs(up_domain))
        np.save('interpolated_B_linear', up_B_lin)

np.save('{}_B_{}'.format(cbc_type, degree), rom.empirical_interpolant.B_matrix)
np.save('{}_fnodes_{}'.format(cbc_type, degree), rom.empirical_interpolant.nodes)

np.save('greedy_params', greedy_samples)

all_samples = rombus._core.mpi.COMM.gather(samples.samples, root=rombus._core.mpi.MAIN_RANK)

if rombus._core.mpi.RANK_IS_MAIN:
    all_samples = [item for sublist in all_samples for item in sublist] 
    np.save('samples', all_samples)
