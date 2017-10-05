import sys
sys.path.append('./src/')
import numpy as np
import networkx as nx
from util import *
import matplotlib.pyplot as plt

n = 100
d_list = [8]

numtrials = 10
numinnertrials = 100

s_upperlim = 16

upper_bound = []
max_lower_bound = []
avg_lin_err = []
avg_opt_err = []

for d_index in range(len(d_list)):
	d = d_list[d_index]
	s_set = list(range(d,s_upperlim))

	print("Starting d = " + str(d))

	for t1 in range(numtrials):
		
		# # Random d-regular graph tests --- expander_ver=0
		# B = getBapprox(n,d,0)
		# assert(all((B[:] == B.T[:]).ravel())) #checking to make sure B is symmetric
		# eigs, eigvecs = np.linalg.eig(d*B)
		# eigs.sort()
		# lambd = max(abs(eigs[0]), abs(eigs[-2]))

		# # Random d-reg bipartite graphs --- expander_ver=1
		# B = getBapprox(n,d,1)
		# a1 = np.dot(np.array(B), np.ones(n))
		# a2 = np.dot(np.array(B.T), np.ones(n))
		# assert(np.linalg.norm(a1 - a2,ord = np.inf) <= 1e-5)
		# U, svals, V = np.linalg.svd(d*B)
		# svals.sort()
		# lambd = max(abs(svals[0]), abs(svals[-2]))

		# Random margulis graphs --- expander_ver=2
		assert(d==8)
		B = getBapprox(n,d,2)
		assert(all((B[:] == B.T[:]).ravel())) #checking to make sure B is symmetric
		eigs, eigvecs = np.linalg.eig(d*B)
		eigs.sort()
		lambd = max(abs(eigs[0]), abs(eigs[-2]))

		for s in s_set:
			upper_bound.append( (d,s,(lambd/d)*np.sqrt(n*s/(n-s))))
			max_lower_bound.append( (d,s,np.sqrt(s/d)))

			lin_err = []
			opt_err = []

			for t2 in range(numinnertrials):
				completed_workers = np.array([False]*n, dtype=bool)
				completed_workers[np.random.permutation(n)[:n-s]] = True

				Alinear = getArowlinear(completed_workers)

				completed_ind_set = [i for i in range(n) if completed_workers[i]]
				Aopt = np.linalg.lstsq(B[completed_ind_set,:].T,np.ones(n))[0]

				lin_err.append(np.linalg.norm(np.dot(Alinear, B)-np.ones((1,n))))
				opt_err.append(np.linalg.norm(np.dot(Aopt,B[completed_ind_set,:])-np.ones((1,n))))
				# avg_lin_err.append((d,s,np.linalg.norm(np.dot(Alinear, B)-np.ones((1,n)))))
				# avg_opt_err.append((d,s,np.linalg.norm(np.dot(Aopt,B[completed_ind_set,:])-np.ones((1,n)))))
			# print("Linear Error = %.3f, Optimal Error = %.3f"%(np.mean(lin_err),np.mean(opt_err)))
			
			avg_lin_err.append((d,s,np.mean(lin_err)))
			avg_opt_err.append((d,s,np.mean(opt_err)))
		print("\t Done with trial "+ str(t1))



# ---- Plotting code
for d_index in range(len(d_list)):

	d = d_list[d_index]
	s_set = list(range(d,s_upperlim))

	fig = plt.figure()
	plt.ylabel('L2-Error')
	plt.xlabel('No. of Stragglers (s) ')
	plt.title('L2-Error vs No. of Stragglers')

	up_set = []
	up_err_set = []

	max_lb_set = []

	lin_set = []
	lin_err_set = []
	opt_set = []
	opt_err_set = []

	for s in s_set:
		up_set.append(np.mean([u[2] for u in upper_bound if u[0]==d and u[1]==s]))
		up_err_set.append(np.std([u[2] for u in upper_bound if u[0]==d and u[1]==s]))

		max_lb_set.append(np.mean([u[2] for u in max_lower_bound if u[0]==d and u[1]==s]))

		lin_set.append(np.mean([u[2] for u in avg_lin_err if u[0]==d and u[1]==s]))
		lin_err_set.append(np.std([u[2] for u in avg_lin_err if u[0]==d and u[1]==s]))

		opt_set.append(np.mean([u[2] for u in avg_opt_err if u[0]==d and u[1]==s]))
		opt_err_set.append(np.std([u[2] for u in avg_opt_err if u[0]==d and u[1]==s]))

	plt.errorbar(s_set, up_set, yerr=up_err_set, linewidth = 1.2, label = 'Upper Bound (theoretical)')
	plt.errorbar(s_set, max_lb_set, linewidth = 1.2, label = 'Lower Bound (worst case)')
	plt.errorbar(s_set, lin_set, yerr=lin_err_set, linewidth = 1.2, label = 'Linear Decoder (avg. case)')
	plt.errorbar(s_set, opt_set, yerr=opt_err_set, linewidth = 1.2, label = 'Optimal Decoder (avg. case)')
	plt.legend(loc='best')
	ax = fig.gca()
	ax.set_xticks(range(d-1,s_upperlim+1), minor = True)
	ax.tick_params(axis = 'both', which = 'minor', labelsize = 0)
	plt.grid(True, which = 'minor', axis = 'x', linestyle = '--', linewidth = '0.4')
	# plt.savefig('randomdreg_d_%d.pdf'%(d))
	# plt.savefig('randomdregbip_d_%d.pdf'%(d))
	plt.savefig('randommargulis_d_%d.pdf'%(d))
	plt.show()
	plt.close()
