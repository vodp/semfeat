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

def trainer_basic(concepts, x_neg, C_, dir_feats, dir_basis, rebalance=False, verbose=False):
	for concept in concepts:
		if os.path.isfile(os.path.join(dir_basis, concept + '.model')):
			continue
		# load features
		if verbose:
			print 'learning #{}#...'.format(concept)
		x_pos, _ = load_svmlight_file(os.path.join(dir_feats, concept + '.train'))
		if not x_pos.shape:
			print '...data loading failed!'
			return
		x = vstack((x_pos, x_neg))
		y = np.array([1]*x_pos.shape[0] + [-1]*x_neg.shape[0])
		f = svm.SVC(C=C_, probability=True, kernel='linear')
		f.fit(x, y.ravel())

		pickle.dump(f, open(os.path.join(dir_basis, concept + '.model'), 'wb'), 1)

# def predictor(concepts, dir_basis, x, out_Q):
# 	outputs = []
# 	for concept in concepts:
# 		print 'predicting {}...'.format(concept)
# 		f = pickle.load(open(os.path.join(dir_basis, concept + '.model'), 'rb'))
# 		y = f.predict_proba(x)
# 		outputs.append(y[:, 1].tolist())
# 	out_Q.put(outputs)

def predictor(concepts, dir_basis, x):
	y = np.empty((x.shape[0], len(concepts)))

	for i, concept in zip(range(len(concepts)), concepts):
		print 'predicting {}...'.format(concept)
		f = pickle.load(open(os.path.join(dir_basis, concept + '.model'), 'rb'))
		y[:, i] = f.predict_proba(x)[:, 0]
	return y

def trainer_adapt(concepts, C_, dir_feats, dir_basis, rebalance=False, verbose=False):
	for concept in concepts:
		# load features
		if verbose:
			print 'learning #{}#...'.format(concept)
		x_pos, _ = load_svmlight_file(os.path.join(dir_feats, concept + '.train'))
		x_neg, _ = load_svmlight_file(os.path.join(dir_feats, concept + '.neg'))
		x = vstack((x_pos, x_neg))
		y = np.array([1]*x_pos.shape[0] + [-1]*x_neg.shape[0])
		f = svm.SVC(C=C_, probability=True, kernel='linear')
		f.fit(x, y.ravel())

		pickle.dump(f, open(os.path.join(dir_basis, concept + '.model'), 'wb'), 1)

def sparsify(x, nnz):
	z = np.zeros(x.shape)
	sorted_ix = np.argsort(x, axis=1)
	entries = sorted_ix[:, -nnz:]
	for i in range(entries.shape[0]):
		for j in range(entries.shape[1]):
			z[i, entries[i, j]] = x[i, entries[i, j]]

	return z

def main(*argv):
	parser = argparse.ArgumentParser()
	parser.add_argument('-m','--mode', help='select the mode: train OR predict', default='train')
	parser.add_argument('-l', '--fn_list_concepts', help='the file containing the list of basis concepts')
	parser.add_argument('-i', '--dir_feats', help='the directory containing feature files for each basis concept')
	parser.add_argument('-f', '--feature_ov', help='the abs. path to the overfeat feature file')
	parser.add_argument('-s', '--signature', help='the output file of semfeat feature')
	parser.add_argument('-o', '--dir_basis', help='the directory containing learned basis classifiers')
	parser.add_argument('-j', '--fn_negative', help='the negative feature file commonly used by all basis')
	parser.add_argument('-C', '--coeff_C', help='the coefficient regularization of SVM', type=float, default=10)
	parser.add_argument('-n', '--nnz', help='the number of non-zeros of semfeat', type=int, default=-1)
	parser.add_argument('-t', '--nprocs', help='the number of parallel processes', default=1, type=int)
	parser.add_argument('expt_name', help='the name for basis learned with specific setting')
	parser.add_argument('-v', '--verbose', help='Talk!', action='store_true')
	args = parser.parse_args()

	# load concepts
	concepts = open(args.fn_list_concepts).read().splitlines()
	print '{} basis identifiers loaded.'.format(len(concepts))

	if args.mode == 'train':
		procs = []
		sz_chunk = int(math.ceil(len(concepts)/float(args.nprocs)))

		if not os.path.isdir(os.path.join(args.dir_basis, args.expt_name)):
			os.mkdir(os.path.join(args.dir_basis, args.expt_name))

		if args.fn_negative:
			if args.verbose:
				print 'loading negative samples...'
			x_neg, _ = load_svmlight_file(args.fn_negative)
			if not x_neg.shape:
				print '...failed!'
				return
			for t in range(args.nprocs):
				p = mp.Process(target=trainer_basic, args=(concepts[sz_chunk*t:sz_chunk*(t+1)], x_neg, args.coeff_C, args.dir_feats, os.path.join(args.dir_basis, args.expt_name), False, args.verbose))
				procs.append(p)
				p.start()
		else:
			print 'training...'
			for t in range(args.nprocs):
				p = mp.Process(target=trainer_adapt, args=(concepts[sz_chunk*t:sz_chunk*(t+1)], args.coeff_C, args.dir_feats, os.path.join(args.dir_basis, args.expt_name), False, args.verbose))
				procs.append(p)
				p.start()

		for p in procs:
			p.join()

	elif args.mode == 'predict':
		print 'loading data...'
		x, _ = load_svmlight_file(args.feature_ov)
		models = []
		y = np.zeros((x.shape[0], len(concepts)))
		i = 0
		
		# procs = []
		# sz_chunk = int(math.ceil(len(concepts)/float(args.nprocs)))
		# out_Q = mp.Queue()

		# for t in range(args.nprocs):
		# 	p = mp.Process(target=predictor, args=(concepts[sz_chunk*i:sz_chunk*(i+1)], os.path.join(args.dir_basis, args.expt_name), x, out_Q))
		# 	procs.append(p)
		# 	p.start()

		# y = []
		# for i in  range(args.nprocs):
		# 	y += out_Q.get()
		y = predictor(concepts, os.path.join(args.dir_basis, args.expt_name), x)
		print 'prediction finished.'

		# for p in procs:
		# 	p.join()

		# sparsification
		z = []
		if args.nnz > 0:
			print 'sparsifying...'
			z = sparsify(y, args.nnz)
		else:
			z = y

		# save it
		dump_svmlight_file(z, [1]*z.shape[0], args.signature)
	else:
		print '!!! Unknown mode. Exited.'
			

if __name__ == '__main__':
	sys.exit(main(*sys.argv))

## training mode
# python /home/phong/semfeat/src/semfeat.py basis_367 -m train -l /home/phong/semfeat/dat/small_groups.txt -i /scratch_global/MUCKE/monsterFeat/overfeatL19_liblinear_L2 -o /home/phong/semfeat/out -j /scratch_global/MUCKE/monsterFeat/bigNegativeClass/negativeSamples_0.3K.libsvm -t 10 -v 

## test mode
# mkdir /home/phong/clsf_voc07/out/basis367_platt
# python /home/phong/semfeat/src/semfeat.py basis_367 -l /home/phong/semfeat/dat/small_groups.txt -o /home/phong/semfeat/out -f /scratch_global/MUCKE/PascalVOC/overfeat_img_test_L2.libsvm -s /home/phong/clsf_voc07/out/basis367_platt/semfeat_C10_neg300_full.test -m predict

# python /home/phong/semfeat/src/semfeat.py basis_367 -l /home/phong/semfeat/dat/small_groups.txt -o /home/phong/semfeat/out -f /scratch_global/MUCKE/PascalVOC/overfeat_img_train_L2.libsvm -s /home/phong/clsf_voc07/out/basis367_platt/semfeat_C10_neg300_full.train -m predict