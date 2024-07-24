import bilby 
import lal
import seaborn as sns
import matplotlib.pyplot as plt
import lalsimulation as lalsim
import numpy as np 
import itertools
from bilby.gw.conversion import earth_motion_time_delay as emtd
from bilby.gw.conversion import (symmetric_mass_ratio_to_mass_ratio, chirp_mass_and_mass_ratio_to_component_masses,
                                    component_masses_to_chirp_mass, component_masses_to_mass_ratio)

# def time_range(tc, m1, m2, chi1, chi2, f_min, fmin_ind, fmax_ind):
#     """
#     Construct an array of times the signal spends between frequency bands ranging in powers of two from f_min to f_max.
#     """
#     times = [emtd(tc, m1,m2,chi1,chi2, 2**(fmin_ind+1))- emtd(tc, m1,m2,chi1,chi2, f_min)]

#     for i in range(fmin_ind+1, fmax_ind):
#         times.append(
#             emtd(tc, m1,m2,chi1,chi2, 2**(i+1)) - emtd(tc, m1,m2,chi1,chi2, 2**i)
#         )
#     times = np.abs(np.array(times))
#     times = 2. ** np.ceil(np.log2(times)).astype(int)
#     return times 


# def frequency_range(f_min, f_max, fmin_ind, fmax_ind, delta_f):
#     """
#     Construct an array of frequencies with different sampling rates.
#     """
#     f_values = np.linspace(f_min, 2**(fmin_ind+1), int((2**(fmin_ind+1)-f_min)/delta_f[0])+1, endpoint=False)
    
#     for i in range(fmin_ind+1, fmax_ind):
#         f_values = np.append(f_values, np.linspace(2**i, 2**(i+1), int((2**i)/delta_f[i-(fmin_ind)])+1, endpoint=False))
    
#     f_values = np.append(f_values, f_max)
#     return f_values

#########################
def down_sample(ini_args, df_max=1.):
    f_min, f_max = float(ini_args.minimum_frequency), float(ini_args.maximum_frequency)
    try:
        priors = bilby.core.prior.PriorDict(ini_args.prior_file)
    except Exception as e:
        priors = inputs.prior_dict

    if 'mass_1' in priors.keys():
        m1_min, m2_min = priors['mass_1'].minimum, priors['mass_2'].minimum
    else:
        q_min, q_max = priors['mass_ratio'].minimum, priors['mass_ratio'].maximum
        mc_min, mc_max = priors['chirp_mass'].minimum, priors['chirp_mass'].maximum
        
        m1_min, _, m2_min, _ = get_component_masses_range(q_min, q_max, mc_min, mc_max)
    
    s1_min, s2_min = priors['chi_1'].minimum, priors['chi_2'].minimum
   
    f_values = banded_sampling(f_min, f_max, m1_min, m2_min, s1_min, s2_min, df_max)
    return f_values 
#########################

def banded_sampling(f_min, f_max, m1_min, m2_min, s1_min, s2_min, df_max):
    """
    Construct an array of frequencies sampled adaptively based on the time spend in each frequency band.
    """

    fmin_ind, fmax_ind = int(np.log2(f_min)), int(np.log2(f_max))

    bounds = [f_min]
    bounds += [2**(i+1) for i in range(fmin_ind, fmax_ind)]

    f_values = np.array([])
    for i in range(len(bounds)-1):
        t1 = lalsim.SimInspiralChirpTimeBound(bounds[i], m1_min*lal.MSUN_SI, m2_min*lal.MSUN_SI, s1_min, s2_min)
        t2 = lalsim.SimInspiralChirpTimeBound(bounds[i+1], m1_min*lal.MSUN_SI, m2_min*lal.MSUN_SI, s1_min, s2_min)

        time = 2. ** np.ceil(np.log2(np.abs(t1-t2))).astype(int)
        deltaF = 1./time

        if deltaF > df_max:
            deltaF = df_max

        f = np.linspace(bounds[i], bounds[i+1], int((bounds[i+1]-bounds[i])/deltaF), endpoint=False)
        f_values = np.append(f_values, f)
    #times = time_range(tc, m1, m2, chi1, chi2, f_min, fmin_ind, fmax_ind)

    #delta_f = 1/times #Some times are negative because of PN expansion inside ISCO. These are masked out anyways.
    #delta_f[delta_f > df_max] = df_max

    #f_values = frequency_range(f_min, f_max, fmin_ind, fmax_ind, delta_f)

    return f_values

#########################

def get_component_masses_range(q_min, q_max, mc_min, mc_max):
    m1_min, _ = chirp_mass_and_mass_ratio_to_component_masses(mc_min, q_max)
    m1_max, _ = chirp_mass_and_mass_ratio_to_component_masses(mc_max, q_min)
    _, m2_min = chirp_mass_and_mass_ratio_to_component_masses(mc_min, q_min)
    _, m2_max = chirp_mass_and_mass_ratio_to_component_masses(mc_max, q_max)

    return m1_min, m1_max, m2_min, m2_max

#########################

def boundary_samples(model):
    minmax = []
    for param in model.params.params:
        minmax.append([param.min, param.max])
   
    combinations = list(itertools.product(*minmax))
    samples = [np.array(combo) for combo in combinations]

    assert len(samples) == 2**(len(model.params.params))
    return samples 

#########################

def random_samples(model, n_random):
    samples = np.ndarray((n_random, len(model.params.params)))
    
    for i in range(n_random):
        samples[i] = get_random_sample(model)

    samples = np.unique(samples, axis=0)
    return samples

#########################

def get_random_sample(model):
    sample = []
    for param in model.params.params:
        if param.name == 'chirp_mass':
            p = 3./5.
            u_sample = np.random.uniform()
            mc_sample = ((param.max**(1-p)-param.min**(1-p))*u_sample + param.min**(1-p))**(1/(1-p))
            sample.append(mc_sample)
        else:
            sample.append(np.random.uniform(param.min, param.max))
    
    return np.array(sample)

#########################

def grid_samples(model, n_random, num_grid=500):
    samples = np.ndarray((n_random, len(model.params.params)))
    
    for i in range(n_random):
        samples[i] = get_grid_sample(model, num_grid)

    samples = np.unique(samples, axis=0)
    return samples

#########################

def get_grid_sample(model, num_grid):
    sample = []
    for param in model.params.params:
        if param.name == 'chirp_mass':
            p = 3./5.
            param_grid = np.linspace(param.min**p, param.max**p, num_grid)**(1./p)
        else:
            param_grid = np.linspace(param.min, param.max, num_grid)
        sample.append(np.random.choice(param_grid))
    
    return np.array(sample)

#########################

def calculate_mismatch(model, samples, B_matrix, fnodes):

    B_matrix = np.load(B_matrix, allow_pickle=True)
    fnodes = np.load(fnodes, allow_pickle=True)

    uniform_f = np.linspace(model.f_min, model.f_max, model.n_f)
    mismatch = [] 
    plot_wf = True 
    for sample in samples:
        sample = model.params.np2param(sample)

        signal_at_nodes = model.compute(sample, fnodes)

        rom = np.dot(signal_at_nodes, B_matrix)

        true = model.compute(sample, uniform_f)

        rom /= np.sqrt(np.vdot(rom,rom).real)    
        true /= np.sqrt(np.vdot(true,true).real)

        mm = 1-np.vdot(rom,true).real
        
        if mm !=0:
            mismatch.append(mm)


    return mismatch


########################

def plot_greedypoints(model, greedy, param1, param2, color='black', label='greedy_points'):
    param1_index = model.params.names.index(param1)
    param2_index = model.params.names.index(param2)

    param1_samples = [row[param1_index] for row in greedy]
    param2_samples = [row[param2_index] for row in greedy] 

    plt.scatter(param1_samples, param2_samples, c=color, alpha=0.5)
    plt.colorbar(label='Mismatch', orientation='horizontal')
    plt.xlabel(param1)
    plt.ylabel(param2)
    plt.savefig('{}.pdf'.format(label))
    plt.clf()
        