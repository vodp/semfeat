# Test the idea of pair-features
import os
import time
import sys
import math
import pickle
import numpy as np 
import argparse
import sklearn.svm as svm
import multiprocessing as mp 
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import dump_svmlight_file
from scipy.sparse import vstack

def subsampling_negative_samples(Xs, i, subsampling=1)	:
	X_neg = []
	for j in range(len(Xs)):
		if j != i: 
			if subsampling < 1 and subsampling > 0:
				ix = np.random.permutation(Xs[j].shape[0])
				ix = ix[:math.floor(subsampling*Xs[j].shape[0])]
				X_neg.append(Xs[j][ix, :])
			else:
				X_neg.append(Xs[j])
				
	X_neg = np.concatenate(X_neg)
	y_neg = np.empty((X_neg.shape[0],))
	y_neg.fill(-1)	
	
	return X_neg, y_neg

def trainer_basis(Xs, concepts, tasks, subsampling, C_, dir_out, verbose=False):
	# load parameters
	for i in tasks:
		if verbose:
			print 'Training #{}#...'.format(concepts[i])
		
		X_neg, y_neg = subsampling_negative_samples(Xs, i, subsampling)
		X_pos = Xs[i]
		y_pos = np.ones((X_pos.shape[0],))
		X = np.concatenate((X_pos, X_neg))
		y = np.concatenate((y_pos, y_neg))

		# centerize data
		#mean_v = np.mean(X, axis=0)
		#X = X - np.tile(mean_v, (X.shape[0], 1))
					
		f = svm.LinearSVC(C=C_, dual=False, fit_intercept=True)
		f.fit(X, y)
		pickle.dump(f, open(os.path.join(dir_out, concepts[i] + '.model'), 'wb'))

def trainer_basis_2(Xs, X_neg, concepts, tasks, subsampling, C_, dir_out, verbose=False):
	# load parameters
	for i in tasks:
		if verbose:
			print 'Training #{}#...'.format(concepts[i])
		
		y_neg = np.empty((X_neg.shape[0], ))
		y_neg.fill(-1)
		X_pos = Xs[i]
		y_pos = np.ones((X_pos.shape[0],))
		X = np.concatenate((X_pos, X_neg))
		y = np.concatenate((y_pos, y_neg))

		# centerize data
		#mean_v = np.mean(X, axis=0)
		#X = X - np.tile(mean_v, (X.shape[0], 1))
					
		f = svm.LinearSVC(C=C_, dual=False, fit_intercept=True)
		f.fit(X, y)
		pickle.dump(f, open(os.path.join(dir_out, concepts[i] + '.model'), 'wb'))

def getW(concepts, dir_out):
	W = np.empty((len(concepts), 4096))
	b = np.empty((len(concepts), ))
	i = 0
	for concept in concepts:
		f = pickle.load(open(os.path.join(dir_out, concept + '.model'), 'rb'))
		W[i, :] = f.coef_
		b[i] = f.intercept_
		i += 1
	return W, b

def predictor(X, W, verbose=False):
	return np.dot(W.T, X)

def load_data(concepts, dir_feats):
	Xs = []
	for concept in concepts:
		X, _ = load_svmlight_file(os.path.join(dir_feats, concept + '.train'))
		# centerize data
		Xs.append(X.toarray())
	return Xs

def main(*argv):
	parser = argparse.ArgumentParser()
	parser.add_argument('-l', '--fn_list_concepts', help='the file containing the list of basis concepts')
	parser.add_argument('-i', '--dir_feats', help='the directory containing feature files for each basis concept')
	parser.add_argument('-o', '--dir_basis', help='the directory containing learned basis classifiers')
	parser.add_argument('-w', '--fn_weights', help='the file name of the weights W. It is stored in the same directory of bases')
	parser.add_argument('-C', '--coeff_C', help='the coefficient regularization of SVM', type=float, default=10)
	parser.add_argument('-t', '--nprocs', help='the number of parallel processes', default=1, type=int)
	parser.add_argument('-s', '--subsampling', type=float, default=1)
	parser.add_argument('expt_name', help='the name for basis learned with specific setting')
	parser.add_argument('-v', '--verbose', help='Talk!', action='store_true')
	parser.add_argument('-j', '--fn_negative', help='the negative feature file commonly used by all basis')
	args = parser.parse_args()

	# load concepts
	concepts = open(args.fn_list_concepts).read().splitlines()
	print '{} basis identifiers loaded.'.format(len(concepts))

	procs = []
	sz_chunk = int(math.ceil(len(concepts)/float(args.nprocs)))
	tasks = range(len(concepts))

	if not os.path.isdir(os.path.join(args.dir_basis, args.expt_name)):
		os.mkdir(os.path.join(args.dir_basis, args.expt_name))

	print 'loading data...'
	Xs = load_data(concepts, args.dir_feats)

	if args.fn_negative:
		X_neg, _ = load_svmlight_file(args.fn_negative)
		X_neg = X_neg.toarray()

	print 'training...'
	for t in range(args.nprocs):
		if not args.fn_negative:
			p = mp.Process(target=trainer_basis, args=(Xs, concepts, tasks[sz_chunk*t:sz_chunk*(t+1)], args.subsampling, args.coeff_C, os.path.join(args.dir_basis, args.expt_name), args.verbose))
		else:
			p = mp.Process(target=trainer_basis_2, args=(Xs, X_neg, concepts, tasks[sz_chunk*t:sz_chunk*(t+1)], args.subsampling, args.coeff_C, os.path.join(args.dir_basis, args.expt_name), args.verbose))
		procs.append(p)
		p.start()
		
	for p in procs:
		p.join()

	print 'saving model...'
	W, b = getW(concepts, os.path.join(args.dir_basis, args.expt_name))
	# save model
	#pickle.dump(W, open(os.path.join(args.dir_basis, args.fn_weights), 'wb'))
	np.savetxt(os.path.join(args.dir_basis, args.expt_name, args.fn_weights + '.weights'), W)
	np.savetxt(os.path.join(args.dir_basis, args.expt_name, args.fn_weights + '.intercepts'), b)

if __name__ == '__main__':
	sys.exit(main(*sys.argv))

## training mode
# python /home/phong/semfeat/src/semfeat.py basis_367 -m train -l /home/phong/semfeat/dat/small_groups.txt -i /scratch_global/MUCKE/monsterFeat/overfeatL19_liblinear_L2 -o /home/phong/semfeat/out -j /scratch_global/MUCKE/monsterFeat/bigNegativeClass/negativeSamples_0.3K.libsvm -t 10 -v 

## test mode
# mkdir /home/phong/clsf_voc07/out/basis367_platt
# python /home/phong/semfeat/src/semfeat.py basis_367 -l /home/phong/semfeat/dat/small_groups.txt -o /home/phong/semfeat/out -f /scratch_global/MUCKE/PascalVOC/overfeat_img_test_L2.libsvm -s /home/phong/clsf_voc07/out/basis367_platt/semfeat_C10_neg300_full.test -m predict

# python /home/phong/semfeat/src/semfeat.py basis_367 -l /home/phong/semfeat/dat/small_groups.txt -o /home/phong/semfeat/out -f /scratch_global/MUCKE/PascalVOC/overfeat_img_train_L2.libsvm -s /home/phong/clsf_voc07/out/basis367_platt/semfeat_C10_neg300_full.train -m predict