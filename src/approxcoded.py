from __future__ import print_function
import sys
import random
from util import *
import os
import numpy as np
import time
from mpi4py import MPI
import scipy.special as sp
import scipy.sparse as sps

def approx_coded_logistic_regression(n_procs, n_samples, n_features, input_dir, n_stragglers, is_real_data, params, trial_num):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    rounds = params['num_itrs']

    n_workers=n_procs-1
    rows_per_worker=n_samples/(n_procs-1)

    B = np.zeros((n_workers,n_workers))

    deg = params['deg']

    if rank==0:
        expander_ver = 0
        B=getBapprox(n_workers,deg, expander_ver)

    B = comm.bcast(B, root=0)

    # Loading the data
    if (rank):

        support_set = [i for i in range(n_workers) if B[rank-1, i]!=0]

        y_current=np.zeros(len(support_set)*rows_per_worker)
        y_current_mod=np.zeros(len(support_set)*rows_per_worker) # Setting up y_current_mod as well 
        y=load_data(input_dir+"label.dat")
        
        if not is_real_data:

            X_current=np.zeros([len(support_set)*rows_per_worker,n_features])
            
            for i in range(len(support_set)):
                X_current[i*rows_per_worker:(i+1)*rows_per_worker,:] = load_data(input_dir+str(support_set[i]+1)+".dat")
                y_current[i*rows_per_worker:(i+1)*rows_per_worker] = y[(support_set[i])*rows_per_worker:(support_set[i]+1)*rows_per_worker]
                y_current_mod[i*rows_per_worker:(i+1)*rows_per_worker] = B[rank-1, i]*y_current[i*rows_per_worker:(i+1)*rows_per_worker]
        else:

            for i in range(len(support_set)):
                                
                if i==0:
                    X_current=load_sparse_csr(input_dir+str(support_set[i]+1))
                else:
                    X_temp = load_sparse_csr(input_dir+str(support_set[i]+1))
                    X_current = sps.vstack((X_current,X_temp))

                y_current[i*rows_per_worker:(i+1)*rows_per_worker] = y[(support_set[i])*rows_per_worker:(support_set[i]+1)*rows_per_worker]
                y_current_mod[i*rows_per_worker:(i+1)*rows_per_worker] = B[rank-1, i]*y_current[i*rows_per_worker:(i+1)*rows_per_worker]

    # Initializing relevant variables
    beta=np.zeros(n_features)
    
    if (rank):

        predy = X_current.dot(beta)
        g = -X_current.T.dot(np.divide(y_current,np.exp(np.multiply(predy,y_current))+1))
        send_req = MPI.Request()
        recv_reqs = []

    else:

        msgBuffers = np.array([np.zeros(n_features) for i in range(n_procs-1)])
    
        g=np.zeros(n_features)

        A_row = np.zeros((1,n_procs-1))

        betaset = np.zeros((rounds, n_features))
        timeset = np.zeros(rounds)
        worker_timeset=np.zeros((rounds, n_procs-1))
        
        request_set = []
        recv_reqs = []
        send_set = []

        cnt_completed = 0
        completed_workers = np.ndarray(n_procs-1,dtype=bool)
        status = MPI.Status()

        eta0=params['learning_rate'] # ----- learning rate
        alpha = params['alpha'] # --- coefficient of l2 regularization
        utemp = np.zeros(n_features) # for accelerated gradient descent

    # Posting all Irecv requests for master and workers
    if (rank):

        for i in range(rounds):
            req = comm.Irecv([beta, MPI.DOUBLE], source=0, tag=i)
            recv_reqs.append(req)
    else:

        for i in range(rounds):
            recv_reqs = []
            for j in range(1,n_procs):
                req = comm.Irecv([msgBuffers[j-1], MPI.DOUBLE], source=j, tag=i)
                recv_reqs.append(req)
            request_set.append(recv_reqs)  

    #######################################################################################################
    comm.Barrier()
    if rank==0:
        print("---- Starting Approx Coded Iterations for " +str(n_stragglers) + " stragglers ----")
        orig_start_time= time.time()

    for i in range(rounds):
    
        if rank == 0:

            if(i%10 == 0):
                print("\t >>> At Iteration %d" %(i))

            A_row[:] = 0
            send_set[:] = []
            completed_workers[:]=False
            cnt_completed = 0

            start_time = time.time()
            
            for l in range(1,n_procs):
                sreq = comm.Isend([beta, MPI.DOUBLE], dest = l, tag = i)
                send_set.append(sreq)            
            
            while cnt_completed < n_procs-1-n_stragglers:
                req_done = MPI.Request.Waitany(request_set[i], status)
                src = status.Get_source()
                worker_timeset[i,src-1]=time.time()-start_time
                request_set[i].pop(req_done)

                cnt_completed += 1
                completed_workers[src-1] = True

            completed_ind_set = [l for l in range(n_procs-1) if completed_workers[l]]

            # ---- optimal decoder
            A_row[0,completed_ind_set] = np.linalg.lstsq(B[completed_ind_set,:].T,np.ones(n_workers))[0]

            # ---- linear time decoder
            # A_row[0, completed_ind_set] = n_workers/(n_workers- n_stragglers)

            g = np.squeeze(np.dot(A_row, msgBuffers))
            
            grad_multiplier = eta0[i]/n_samples
            # ---- update step for gradient descent
            np.subtract((1-2*alpha*eta0[i])*beta , grad_multiplier*g, out=beta)

            # ---- updates for accelerated gradient descent
            # theta = 2.0/(i+2.0)
            # ytemp = (1-theta)*beta + theta*utemp
            # betatemp = ytemp - grad_multiplier*g - (2*alpha*eta0[i])*beta
            # utemp = beta + (betatemp-beta)*(1/theta)
            # beta[:] = betatemp

            timeset[i] = time.time() - start_time
            betaset[i,:] = beta

            ind_set = [l for l in range(n_procs-1) if not completed_workers[l]]
            for l in ind_set:
                worker_timeset[i,l]=-1

        else:
            recv_reqs[i].Wait()

            sendTestBuf = send_req.test()
            if not sendTestBuf[0]:
                send_req.Cancel()
                #print("Worker " + str(rank) + " cancelled send request for Iteration " + str(i))

            predy = X_current.dot(beta)
            g = X_current.T.dot(np.divide(y_current_mod,np.exp(np.multiply(predy,y_current))+1))
            g *= -1
            send_req = comm.Isend([g, MPI.DOUBLE], dest=0, tag=i)

    #####################################################################################################
    comm.Barrier()
    if rank==0:
        elapsed_time= time.time() - orig_start_time
        print ("Total Time Elapsed: %.3f" %(elapsed_time))
        # Load all training data
        if not is_real_data:
            X_train = load_data(input_dir+"1.dat")
            print(">> Loaded 1")
            for j in range(2,n_procs-1):
                X_temp = load_data(input_dir+str(j)+".dat")
                X_train = np.vstack((X_train, X_temp))
                print(">> Loaded "+str(j))
        else:
            X_train = load_sparse_csr(input_dir+"1")
            for j in range(2,n_procs-1):
                X_temp = load_sparse_csr(input_dir+str(j))
                X_train = sps.vstack((X_train, X_temp))

        y_train = load_data(input_dir+"label.dat")
        y_train = y_train[0:X_train.shape[0]]

        # Load all testing data
        y_test = load_data(input_dir + "label_test.dat")
        if not is_real_data:
            X_test = load_data(input_dir+"test_data.dat")
        else:
            X_test = load_sparse_csr(input_dir+"test_data")

        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        training_loss = np.zeros(rounds)
        testing_loss = np.zeros(rounds)
        auc_loss = np.zeros(rounds)

        from sklearn.metrics import roc_curve, auc

        for i in range(rounds):
            beta = np.squeeze(betaset[i,:])
            predy_train = X_train.dot(beta)
            predy_test = X_test.dot(beta)
            training_loss[i] = calculate_loss(y_train, predy_train, n_train)
            testing_loss[i] = calculate_loss(y_test, predy_test, n_test)
            fpr, tpr, thresholds = roc_curve(y_test,predy_test, pos_label=1)
            auc_loss[i] = auc(fpr,tpr)
            print("Iteration %d: Train Loss = %5.3f, Test Loss = %5.3f, AUC = %5.3f, Total time taken =%5.3f"%(i, training_loss[i], testing_loss[i], auc_loss[i], timeset[i]))
        
        output_dir = input_dir + "new_results/trial_"+str(trial_num)+"/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        save_vector(training_loss, output_dir+"approxcoded_%d_%d_%d_training_loss.dat"%(n_workers,n_stragglers,deg))
        save_vector(testing_loss, output_dir+"approxcoded_%d_%d_%d_testing_loss.dat"%(n_workers,n_stragglers,deg))
        save_vector(auc_loss, output_dir+"approxcoded_%d_%d_%d_auc.dat"%(n_workers,n_stragglers,deg))
        save_vector(timeset, output_dir+"approxcoded_%d_%d_%d_timeset.dat"%(n_workers,n_stragglers,deg))
        save_matrix(worker_timeset, output_dir+"approxcoded_%d_%d_%d_worker_timeset.dat"%(n_workers,n_stragglers,deg))
        print(">>> Done")

    comm.Barrier()