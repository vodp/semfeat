import os
import time
import sys
import math
import pickle
import numpy as np 
import argparse
from sklearn.svm import LinearSVC
import multiprocessing as mp 
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import dump_svmlight_file
from sklearn.metrics import average_precision_score

def solver(PHI, X, y, C_, alpha_, verbose=False):

	Lambda = np.random.rand(PHI.shape[1])
	
	w_prev = np.zeros((PHI.shape[1], ))
	diff = 1e10
	num_iters = 0
	while diff < epsilon_:

		start = time.time()
		Z = np.dot(np.diag(Lambda), np.dot(PHI.T, X))

		clf = LinearSVC(C=C_, dual=False)
		clf.fit(Z.T, y)

		w = clf.coef_

		Z = np.dot(np.diag(w), np.dot(PHI.T, X))
		clf2 = LinearSVC(C=C_/alpha_, dual=False, penalty='l1')
		clf2.fit(Z, y)

		stop = time.time()

		Lambda = clf2.coef_

		diff = np.sum(np.abs(w_prev - w))
		num_iters += 1

		if verbose:
			print 'iter#{0}\t time_elapsed={1}\t diff={2}\t nnz={3}'.format(num_iters, stop - start, diff, np.count_nonzero(Lambda))
	return w, Lambda

def training(X, PHI, concepts, task, args):
	for i in task:
		y, difficult_cases = load_label(os.path.join(args.dir_label, concepts[i] + '.txt'))
		X1 = np.delete(X, difficult_cases, axis=0)

		if args.verbose:
			print 'learing #{}#...'.format(concepts[i])
		w, Lambda = solver(PHI, X1, y, args.C, args.alpha, args.verbose)

		np.savetxt(os.path.join(args.dir_res, concepts[i] + '.weights'), w)
		np.savetxt(os.path.join(args.dir_res, concepts[i] + '.lambda'), Lambda)

def test(X, PHI, concepts, args):
	for concept in concepts:
		y, difficult_cases = load_label(os.path.join(args.dir_label, concept + '.txt'))
		X1 = np.delete(X, difficult_cases, axis=0)

		print 'loading #{}#...'.format(concept)
		w = np.loadtxt(os.path.join(args.dir_res, concept + '.weights'))
		Lambda = np.loadtxt(os.path.join(args.dir_res, concept + '.lambda'))

		resp = predictor(PHI, X1, w, Lambda)
		pickle.dump([resp, y], open(os.path.join(args.dir_res, concept + '.score')))

def predictor(PHI, X, w, Lambda, verbose=False):
	return np.dot(np.dot(PHI, np.dot(np.diag(Lambda), w)), X)

def evaluate(concepts, args):
	aps = []
	f = open(os.path.join(args.res_dir, args.output), 'wt')
	for concept in concepts:
		fn_score = os.path.join(args.dir_res, concept + '.score')
		y_score, y_true = pickle.load(open(fn_score, 'rb'))

		ap = average_precision_score(y_true, y_score)
		f.write('%s\t\tAP=%2.5f\n' % (concept, ap))
		aps.append(ap)
	
	mean_ap = sum(aps)/float(len(aps))
	f.write('Summary:\t\tmAP=%2.5f\n' % (mean_ap))
	return mean_ap

def load_label(fn_label):
	y = []
	difficult_cases = []
	i = 0
	with open(os.path.join(fn_label)) as f:
		for line in f:
			raw_label = int(line.strip().split()[1])
			if raw_label == 0: 
				difficult_cases.append(i)
			else:
				y.append(raw_label)
			i += 1
	return np.array(y), difficult_cases

def main(*argv):
	parser = argparse.ArgumentParser()
	parser.add_argument('concepts', help='indicating either: i) the full path to the file storing the list of NUSWIDE concepts used for training, ii) list of concepts need to be tested')
	parser.add_argument('fn_data', help='the feature file name')
	parser.add_argument('fn_basis_set', help='the path to the basis file')
	parser.add_argument('dir_label', help='the label file name')
	parser.add_argument('-r', '--dir_res', help='the directory storing learning outputs. In #train# mode, it automatically create a sub-directory under this directory in order to dump SVM model files.', default='.')
	parser.add_argument('-m', '--mode', help='the training mode: either train, test, tune or arbitrary combinations among them; for instance tune+train, train+test.')
	parser.add_argument('-c', '--C', help='the regularization coefficient C of SVM', type=float, default=-1)	
	parser.add_argument('-a', '--alpha', help='the sparsity coefficient', type=float, default=0.1)	
	parser.add_argument('-n', '--name', help='The naming for this experiment. By default, the name will be automatically generated according to some settings of the problem.', default='default')
	parser.add_argument('-v', '--verbose', action='store_true')
	parser.add_argument('-t', help='the number of processes for parallel training', type=int, default=1)
	args = parser.parse_args()

	print 'loading concepts...'
	concepts = open(args.concepts).read().splitlines()
	
	print 'loading data...'
	X, _ = load_svmlight_file(args.fn_data)
	PHI = np.loadtxt(args.fn_basis_set)

	if args.t == -1:
		args.t = range(len(concepts))

	if args.mode == 'train':
		print '.:TRAINING:.'
		procs = []
		sz_chunk = int(math.ceil(len(concepts)/float(args.t)))
		tasks = range(len(concepts))
		for proc in args.t:
			p = mp.Process(target=training, args=(X, PHI, concepts, tasks[sz_chunk*t:sz_chunk*(t+1)], args))
			procs.append(p)
			p.start()

		for p in procs:
			p.join()

	elif args.mode == 'test':
		print '.:EVALUATION:.'
		test(X, PHI, concepts, args)
		evaluate(concepts, args)

if __name__ == '__main__':
	sys.exit(main(*sys.argv))