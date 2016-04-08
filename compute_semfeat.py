import sys
import math
import pickle
import numpy as np 
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import dump_svmlight_file


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
	parser.add_argument('fn_train', help='the training feature file')
	parser.add_argument('fout_train', help='the normalized training feature file')
	parser.add_argument('fn_test', help='the test feature file')
	parser.add_argument('fout_test', help='the normalized test feature file')
	parser.add_argument('fn_weights')
	parser.add_argument('-b', '--fn_intercepts', default=[])
	parser.add_argument('-n', '--nnz', help='the number of non-zeros retained', type=int, default=-1)
	
	args = parser.parse_args()

	print 'loading model...'
	W = np.loadtxt(args.fn_weights)
	if args.fn_intercepts:
		b = np.loadtxt(args.fn_intercepts)

	print 'loading training data...'
	X, _ = load_svmlight_file(args.fn_train)
	X = X.toarray()

	print 'computing scores...'
	if args.fn_intercepts:
		Y = np.dot(W, X.T) + np.tile(b, (X.shape[0], 1)).T
	else:
		Y = np.dot(W, X.T)
	Y = Y.T

	print 'normalizing...'
	scaler = MinMaxScaler()
	Y = scaler.fit_transform(Y)

	if args.nnz > 0:
		print 'sparsifying...'
		Y = sparsify(Y, args.nnz)

	print 'writing seamfeat[training data]...'
	dump_svmlight_file(Y, [1]*Y.shape[0], args.fout_train)

	##
	print 'loading test data...'
	Xt, _ = load_svmlight_file(args.fn_test)
	Xt = Xt.toarray()

	print 'computing scores...'
	if args.fn_intercepts:
		Y = np.dot(W, Xt.T) + np.tile(b, (Xt.shape[0], 1)).T
	else:
		Y = np.dot(W, Xt.T)

	Y = Y.T

	print 'normalizing...'
	Y = scaler.transform(Y)

	if args.nnz > 0:
		print 'sparsifying...'
		Y = sparsify(Y, args.nnz)

	print 'writing seamfeat[test data]...'
	dump_svmlight_file(Y, [1]*Y.shape[0], args.fout_test)

if __name__ == '__main__':
	sys.exit(main(*sys.argv))	

