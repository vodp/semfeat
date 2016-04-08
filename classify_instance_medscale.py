# evaluate NUS-WIDE classification problem with 81 concepts
# evaluation is based on APs and mAP

# Examples:

import os
import sys
import cPickle as pickle
import math
import numpy as np 
from sklearn import cross_validation as cv
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import MinMaxScaler
import argparse
from collections import OrderedDict
import operator

sys.path.append(r'/home/phong/extlibs/liblinear-1.94/python')
sys.path.append(r'/home/phong/extlibs/liblinear-1.94')
from liblinearutil import *

def normalize_L2(X):
	return X/np.tile(np.reshape(np.sqrt((X*X).sum(axis=1)), (X.shape[0], 1)), (1, X.shape[1]))


def find_params_linear(y, X, k=3, rebalance=False):
	C = [2**i for i in np.arange(-10, 11, 1).tolist()]
	best_C = 0 
	acc_max = 0

	# generate K-folds
	k_fold = cv.KFold(len(y), n_folds=k)
	for c in C:
		params = parameter('-s {0} -c {1} -q'.format(2, c))
		acc = []
		for tn, tt in k_fold:
			y_tn = [y[i] for i in range(len(tn)) if tn[i]]
			X_tn = [X[i] for i in range(len(tn)) if tn[i]]

			y_tt = [y[i] for i in range(len(tt)) if tt[i]]
			X_tt = [X[i] for i in range(len(tt)) if tt[i]]

			prob = problem(y_tn, X_tn)
			f = train(prob, params)
			_, _, score = predict(y_tt, X_tt, f)
			score = [z for x in score for z in x]
			acc.append(average_precision_score(y_tt, score))
		if np.mean(acc) > acc_max:
			acc_max = np.mean(acc)
			best_C = c
	return best_C, acc_max

def tuner(X, concepts, tasks, labels, subsampling, dir_res, rebalance_, dimensions, verbose=False):
	results = {}
	
	for i in tasks:
		if verbose:
			print 'Tunning #{}#: prepare data...'.format(concepts[i]),
			sys.stdout.flush()
	
		y_gt = labels[concepts[i]]
		ix = np.random.permutation(len(y_gt))
		X1 = [X[j] for j in ix]
		y1 = [y_gt[j] for j in ix]
		
		params = {}
		if verbose:
			print 'running 3-fold CV...',
			sys.stdout.flush()
		best_C, acc = find_params_linear(y1, X1, rebalance=rebalance_)
		params['C'] = best_C
		if verbose:
			print '...C=%4.4f, \t AP=%2.2f' % (best_C, acc)
		pickle.dump(params, open(os.path.join(dir_res, concepts[i] + '.params'), 'wb'), 1)

def trainer(X, concepts, tasks, labels, subsampling, C_, rebalance, dir_out, dimensions, verbose=False):

	for i in tasks:
		params = []
		file_out = os.path.join(dir_out, concepts[i] + '.params')
		if C_ == -1 and os.path.isfile(file_out):
			params = pickle.load(open(file_out))
			if verbose:
				print 'Params file found and loaded.'
		elif C_ == -1:
			if verbose:
				print 'Params file not found. Use default setting or specified arguments'
			C_ = 1
		else:
			print 'Set C={}'.format(C_)

		if verbose:
			print 'Training #{}#: prepare data...'.format(concepts[i]),
			sys.stdout.flush()

		# switch ON for samples belonging to task i-th
		y_gt = labels[concepts[i]]
		if verbose:
			print 'training...',
			sys.stdout.flush()
		if params:
			C_ = params['C']
			
		if dimensions > len(y_gt):
			params = parameter('-s {0} -c {1} -q'.format(1, C_))
		else:
			params = parameter('-s {0} -c {1} -q'.format(2, C_))
		#print(len(y_gt))
		#print(len(X))
		prob = problem(y_gt, X)
		f = train(prob, params)
		
		save_model(os.path.join(dir_out, concepts[i] + '.model'), f)
		_, _, p_vals = predict(y_gt, X, f)
		score = [z for x in p_vals for z in x]
		ap = average_precision_score(y_gt, score)
		if verbose:
			print 'AP=%2.2f' % (ap)

def tester(X, concepts, tasks, labels, dir_model, verbose=False):
	for i in tasks:
		if verbose:
			print 'Testing #{}#...'.format(concepts[i]),
			sys.stdout.flush()
		
		f = load_model(os.path.join(dir_model, concepts[i] + '.model'))
		
		y_gt = labels[concepts[i]]
		
		_, _, p_vals = predict(y_gt, X, f)
		scores = [z for x in p_vals for z in x]
		ap = average_precision_score(y_gt, scores)
		if verbose:
			print 'AP=%2.2f' % (ap)
		# write result down
		pickle.dump([scores, y_gt], open(os.path.join(dir_model, concepts[i] + '.score'), 'wb'), 1)
		
def main(*argv):
	parser = argparse.ArgumentParser()
	parser.add_argument('-v', '--verbose', action='store_true')
	parser.add_argument('concepts', help='indicating either: i) the full path to the file storing the list of NUSWIDE concepts used for training, ii) list of concepts need to be tested')
	parser.add_argument('-i', '--dir_feat', help='the directory storing image descriptors', default='.')
	parser.add_argument('-l', '--dir_labels', help='the directory storing image label', default='.')
	parser.add_argument('fn_traindata', help='the feature file name of positive data')
	parser.add_argument('fn_testdata', help='the feature file name of negative data')
	parser.add_argument('-r', '--dir_res', help='the directory storing learning outputs. In #train# mode, it automatically create a sub-directory under this directory in order to dump SVM model files.', default='.')
	parser.add_argument('-s', '--subsampling', help='the subsampling rate of the negative samples in the one-vs-rest classification mode. It is also the subsampling rate applied in parameter tuning in order to save time. When you want all images of 80 concepts to be negative samples, just indicate 1 or neglect it.', type=float, default=1)
	parser.add_argument('-m', '--mode', help='the training mode: either train, test, tune or arbitrary combinations among them; for instance tune+train, train+test.')
	parser.add_argument('-c', '--C', help='the regularization coefficient C of SVM', type=float, default=-1)
	parser.add_argument('-b', '--balance', help='whether re-balancing data in training', action='store_true')
	parser.add_argument('-t', help='the specified task index(es) selected out of the concept list to run. See the qsub_generator.py for clarification.', nargs='+', type=int)
	parser.add_argument('-f', '--feature', help='The format of the feature files, either #svmlight# or #binary#', default='binary')
	parser.add_argument('-z', '--minmax_norm', help='to normalize min-max on semfeat features', action='store_true')

	parser.add_argument('-d', '--dimensions', help='The number of feature dimension', type=int, default=20000)
	parser.add_argument('-n', '--name', help='The naming for this experiment. By default, the name will be automatically generated according to some settings of the problem.')
	args = parser.parse_args()

	concepts = open(args.concepts).read().splitlines()
	modes = args.mode.strip().split('+')

	# initialize parallel training on concepts
	print '.:Start doing stuffs:.'
	#if 'train' in modes or 'tune' in modes:
	if not args.name:
		subdir = 'lin-%s-%1.1f' % (args.feature, args.subsampling)
		dir_expt = os.path.join(args.dir_res, subdir)
		if not os.path.isdir(dir_expt):
			os.mkdir(dir_expt)
	else:
		dir_expt = os.path.join(args.dir_res, args.name)
		if not os.path.isdir(dir_expt):
			os.mkdir(dir_expt)

	# load the whole data
	X_tn = []
	X_tt = []
	print 'Loading data...'
	if 'train' in modes or 'tune' in modes:
		if args.feature == 'binary':
			X_tn = pickle.load(open(os.path.join(args.dir_feat, args.fn_traindata), 'rb'))
		else:
			X_tn, _ = load_svmlight_file(os.path.join(args.dir_feat, args.fn_traindata), n_features=args.dimensions)

		# #X_tn = normalize_L2(X_tn.toarray())
		# scaler = MinMaxScaler()
		# X_tn = scaler.fit_transform(X_tn.toarray())

		# save the scaler
		pickle.dump(scaler, open(os.path.join(dir_expt, 'scaler'), 'wb'), 1)

		# load labels
		labels_tn = {}
		for concept in concepts:
			with open(os.path.join(args.dir_labels, concept + '_trainval.txt')) as f:
				labels = []
				for line in f:
					lbl = int(line.strip().split()[1])
					if lbl == 0:
						labels.append(-1)
					else:
						labels.append(lbl)
				labels_tn[concept] = labels

	if 'test' in modes:
		if args.feature == 'binary':
			X_tt = pickle.load(open(os.path.join(args.dir_feat, args.fn_testdata), 'rb'))
		else:
			X_tt, _ = load_svmlight_file(os.path.join(args.dir_feat, args.fn_testdata), n_features=args.dimensions)

		# #X_tt = normalize_L2(X_tt.toarray())
		# if os.path.isfile(os.path.join(dir_expt, 'scaler')):
		# 	scaler = pickle.load(open(os.path.join(dir_expt, 'scaler'), 'rb'))	
		# 	X_tt = scaler.transform(X_tt.toarray())		

		# load labels
		labels_tt = {}
		for concept in concepts:
			with open(os.path.join(args.dir_labels, concept + '_test.txt')) as f:
				labels = []
				for line in f:
					lbl = int(line.strip().split()[1])
					if lbl == 0:
						labels.append(-1)
					else:
						labels.append(lbl)
				labels_tt[concept] = labels

	X_tn = X_tn.tolist()
	X_tt = X_tt.tolist()

	if 'tune' in modes:
		tuner(X_tn, concepts, args.t, labels_tn, args.subsampling, dir_expt, args.balance, args.dimensions, args.verbose)

	if 'train' in modes:
		trainer(X_tn, concepts, args.t, labels_tn, args.subsampling, args.C, args.balance, dir_expt, args.dimensions, args.verbose)

	if 'test' in modes:
		tester(X_tt, concepts, args.t, labels_tt, dir_expt, args.verbose)

	
if __name__ == '__main__':
	sys.exit(main(*sys.argv))
