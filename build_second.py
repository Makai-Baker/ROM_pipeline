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

if ini_args.frequency_domain_source_model == "lal_binary_black_hole":
    cbc_type = 'bbh'
else:
    cbc_type = 'bns'

degrees = ['quadratic']

if adapt_sampling:
    degrees = ['linear', 'quadratic']

tol = 1e-12

for degree in degrees:   
    #model = RombusModel.load('{}_model_{}:GWModel'.format(cbc_type,degree))
    model = RombusModel.load('model:GWModel')

    f_min, f_max, duration = float(ini_args.minimum_frequency), float(ini_args.maximum_frequency), float(ini_args.duration)

    uniform_f = np.linspace(f_min, f_max, int((f_max - f_min)*duration) + 1 )

    model.domain = uniform_f 
    model.n_domain = len(uniform_f)
    up_samples = Samples(model=model, n_random=1)

    up_samples._add_from_file('greedy_params.npy')
 
    rom = ReducedOrderModel(model, up_samples).build(do_step=None, tol=tol)
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