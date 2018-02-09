from __future__ import print_function
import sys
sys.path.append('./src/')
from naive import *
from coded import *
from approxcoded import *
from avoidstragg import *
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if len(sys.argv) != 10:
    print("Usage: python main.py n_procs n_rows n_cols input_dir is_real dataset is_coded n_stragglers coded_ver")
    sys.exit(0)

n_procs, n_rows, n_cols, input_dir, is_real, dataset, \
    is_coded, n_stragglers, coded_ver, trial_num = [x for x in sys.argv[1:]]
n_procs, n_rows, n_cols, is_real, is_coded, n_stragglers, coded_ver, trial_num = \
    int(n_procs), int(n_rows), int(n_cols), int(is_real), int(is_coded), int(n_stragglers), int(coded_ver), int(trial_num)
input_dir = input_dir+"/" if not input_dir[-1] == "/" else input_dir


# ---- Modifiable parameters
num_itrs = 100  # Number of iterations

alpha = 1.0/n_rows  # sometimes we used 0.0001 # --- coefficient of l2 regularization

learning_rate_schedule = 10.0*np.ones(num_itrs)
# eta0 = 10.0
# t0 = 90.0
# learning_rate_schedule = [eta0*t0/(i + t0) for i in range(1, num_itrs+1)]

# -------------------------------------------------------------------------------------------------------------------------------

params = dict()
params['num_itrs'] = num_itrs
params['alpha'] = alpha
params['learning_rate'] = learning_rate_schedule

if not size == n_procs:
    print("Number of processers doesn't match!")
    sys.exit(0)

if not is_real:
    dataset = "artificial-data/" + str(n_rows) + "x" + str(n_cols)

if is_coded:
    
    if coded_ver == 0:  # cyclic coded
        learning_rate_schedule = 10.0 * np.ones(num_itrs)
        params['learning_rate'] = learning_rate_schedule
        coded_logistic_regression(n_procs, n_rows, n_cols, input_dir + dataset + "/" + str(n_procs-1) + "/", n_stragglers, is_real, params, trial_num)
        
    elif coded_ver == 1:  # approx coded
        approx_coded_logistic_regression(n_procs, n_rows, n_cols, input_dir + dataset + "/" + str(n_procs-1) + "/", n_stragglers, is_real, params, trial_num)

    elif coded_ver == 2:  # ignore stragglers
        avoidstragg_logistic_regression(n_procs, n_rows, n_cols, input_dir + dataset + "/" + str(n_procs-1) + "/", n_stragglers, is_real, params, trial_num)
else:
    naive_logistic_regression(n_procs, n_rows, n_cols, input_dir + dataset + "/" + str(n_procs-1) + "/", is_real, params, trial_num)

comm.Barrier()
MPI.Finalize()
