import os
import sys
import numpy as np
import operator
from collections import OrderedDict
import cPickle as pickle 
import argparse
import multiprocessing as mp 
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import dump_svmlight_file
import time 
# def read_semfeat(fn):
# 	X = []
# 	with open(fn, 'rt') as f:
# 		for line in f:
# 			feature = {}
# 			pairs = line.strip().split()
# 			max_v = 0.0
# 			min_v = 1.0
# 			for pair in pairs:
# 				ix, v = pair.strip().split(':')
# 				v = float(v)
# 				feature[int(ix)] = v
# 				if v > max_v:
# 					max_v = v
# 				if v < min_v:
# 					min_v = v 

# 			feature = OrderedDict(sorted(feature.items(), key=operator.itemgetter(1), reverse=True))
# 			vlist = feature.values()#[:10]
# 			klist = feature.keys()#[:10]
# 			feature = {k:(v - min_v)/(max_v - min_v) for (k,v) in zip(klist, vlist)}
# 			#feature = {ix:(v - min_v)/(max_v - min_v) for ix, v in feature.iteritems()}
# 			# sort them in ascending order of index
# 			feature = OrderedDict(sorted(feature.items(), key=operator.itemgetter(0)))

# 			X.append(feature)

# 	return X

def find_minmax(fn_in, ndim):
	min_dict = {k:1 for k in range(ndim)}
	max_dict = {k:0 for k in range(ndim)}
	with open(fn_in, 'rt') as f:
		for line in f:
#			start = time.time()
			feature = {}
			pairs = line.strip().split()
			
			for pair in pairs:
				ix, v = pair.strip().split(':')
				ix = int(ix)
				v = float(v)
				
				if min_dict[ix] > v:
					min_dict[ix] = v

				if max_dict[ix] < v:
					max_dict[ix] = v
	f.close()
	return min_dict, max_dict

def fuck_semfeat(fn_in, fn_out, min_dict, max_dict, nnz):
	fin = open(fn_in, 'rt')
	fout = open(fn_out, 'wt')

	coeffs = [1.0/(max_dict[i] - min_dict[i]) for i in range(len(min_dict))]

	i = 0
	for line in fin:
		print(i)
		i+=1
		feature = {}
		pairs = line.strip().split()
			
		for pair in pairs:
			ix, v = pair.strip().split(':')
			ix = int(ix)
			feature[ix] = (float(v) - min_dict[ix])*coeffs[ix]
		
		# sort them in descending order of score
		feature = OrderedDict(sorted(feature.items(), key=operator.itemgetter(1), reverse=True))

		keys = feature.keys()[:nnz]
		values = feature.values()[:nnz]

		feature = {k:v for k, v in zip(keys, values)}

		feature = OrderedDict(sorted(feature.items(), key=operator.itemgetter(0)))
		
		fout.write('1 ')
		for k, v in zip(feature.keys(), feature.values()):
				fout.write('%d:%f ' % (k, v))
		fout.write('\n')
	fin.close()
	fout.close()

def kick_semfeat(fn_in, fn_out, nnz):
	fin = open(fn_in, 'rt')
	fout = open(fn_out, 'wt')

	i = 0
	for line in fin:
		print(i)
		i+=1
		feature = {}
		pairs = line.strip().split()
			
		for pair in pairs:
			ix, v = pair.strip().split(':')
			ix = int(ix)
			feature[ix] = float(v)
		
		# sort them in descending order of score
		feature = OrderedDict(sorted(feature.items(), key=operator.itemgetter(1), reverse=True))

		min_v = min(feature.values())
		max_v = max(feature.values())

		coeff = 1.0/(max_v - min_v)

		keys = feature.keys()[:nnz]
		values = feature.values()[:nnz]

		values = [(v - min_v)*coeff for v in values]

		feature = {k:v for k, v in zip(keys, values)}

		feature = OrderedDict(sorted(feature.items(), key=operator.itemgetter(0)))
		
		fout.write('1 ')
		for k, v in zip(feature.keys(), feature.values()):
				fout.write('%d:%f ' % (k, v))
		fout.write('\n')
	fin.close()
	fout.close()


# def read_semfeat(fn):
# 	count = 0
# 	X = []
# 	with open(fn, 'rt') as f:
# 		for line in f:
# #			start = time.time()
# 			feature = {}
# 			pairs = line.strip().split()
			
# 			for pair in pairs:
# 				ix, v = pair.strip().split(':')
# 				feature[int(ix)] = float(v)
				
# 			# sort them in ascending order of index
# 			feature = OrderedDict(sorted(feature.items(), key=operator.itemgetter(0)))
# 			X.append(feature)
# #			end = time.time()
# 			count += 1
# 			print (count)
# 	return X	

# WARNING: just use when x is a dense overfeat feature loaded by read_semfeat
# def dict2array(x):
# 	z = []
# 	for feature in x:
# 		z.append(np.array(feature.values()))
# 	return np.array(z)

# def dump_semfeat(X, fn, nnz, y=[]):
# 	if not y:
# 		y = [1]*len(X)
	
# 	with open(fn, 'wt') as f:
# 		for feat, label in zip(X, y):
			
# 			f.write('%d ' % (label))
# 			idx = feat.keys()
# 			v = feat.values()
# 			for idx, v in zip(feat.keys(), feat.values()):
# 				f.write('%d:%f ' % (idx, v))
# 			f.write('\n')
# 	f.close()

# def sparsify(x, nnz):
# 	z = np.zeros(x.shape)
# 	sorted_ix = np.argsort(x, axis=1)
# 	entries = sorted_ix[:, -nnz:]
# 	for i in range(entries.shape[0]):
# 		for j in range(entries.shape[1]):
# 			z[i, entries[i, j]] = x[i, entries[i, j]]

# 	return z

def main(*argv):
	parser = argparse.ArgumentParser()
	parser.add_argument('fn_train', help='the training feature file')
	parser.add_argument('fout_train', help='the output training feature file')
	parser.add_argument('fn_test', help='the training feature file')
	parser.add_argument('fout_test', help='the output training feature file')
	parser.add_argument('-n', '--nnz', help='the number of non-zeros retained', type=int, default=-1)
	parser.add_argument('-m', '--mode', help='fuck or kick?', default='fuck')

	args = parser.parse_args()

	if args.mode == 'fuck':
		print 'finding min, max...'
		min_dict, max_dict = find_minmax(args.fn_train, 30000)

		print 'converting {}'.format(args.fn_train)
		p1 = mp.Process(target=fuck_semfeat, args=(args.fn_train, args.fout_train, min_dict, max_dict, args.nnz))
		p1.start()
		#fuck_semfeat(args.fn_train, args.fout_train, min_dict, max_dict, args.nnz)

		print 'converting {}'.format(args.fn_test)
		p2 = mp.Process(target=fuck_semfeat, args=(args.fn_test, args.fout_test, min_dict, max_dict, args.nnz))
		p2.start()

		p1.join()
		p2.join()
	elif args.mode == 'kick':
		print 'converting {}'.format(args.fn_train)
		p1 = mp.Process(target=kick_semfeat, args=(args.fn_train, args.fout_train, args.nnz))
		p1.start()
		#fuck_semfeat(args.fn_train, args.fout_train, min_dict, max_dict, args.nnz)

		print 'converting {}'.format(args.fn_test)
		p2 = mp.Process(target=kick_semfeat, args=(args.fn_test, args.fout_test, args.nnz))
		p2.start()

		p1.join()
		p2.join()
	#fuck_semfeat(args.fn_test, args.fout_test, min_dict, max_dict, args.nnz)
	# print 'loading training data...'
	# x_tn = read_semfeat(args.fn_train)
	# x_tn = dict2array(x_tn)

	# print 'loading test data...'
	# x_tt = read_semfeat(args.fn_test)
	# x_tt = dict2array(x_tt)

	# print 'normalizing...'
	# scaler = MinMaxScaler()
	# x_tn = scaler.fit_transform(x_tn)
	# x_tt = scaler.transform(x_tt)

	# if args.nnz > 0:
	# 	print 'sparsifying...'
	# 	x_tn = sparsify(x_tn, args.nnz)
	# 	x_tt = sparsify(x_tt, args.nnz)

	# print 'dumping...'
	# dump_svmlight_file(x_tn, [1]*x_tn.shape[0], args.fout_train)
	# dump_svmlight_file(x_tt, [1]*x_tt.shape[0], args.fout_test)

if __name__ == '__main__':
	sys.exit(main(*sys.argv))
##
# fn_test = '/home/phong/imdbs/VOC07/groups_367_sparse_10/semfeat_img_test_sparse_10_c_10_sknn_80_neg_300.fake.libsvm'
# fn_train = '/home/phong/imdbs/VOC07/groups_367_sparse_10/semfeat_img_train_sparse_10_c_10_sknn_80_neg_300.fake.libsvm'
# DIR_FEAT = '/home/phong/imdbs/VOC07/groups_367_sparse_10'

# x_tn = read_semfeat(fn_train)
# x_tt = read_semfeat(fn_test)

# dump_semfeat(x_tn, os.path.join(DIR_FEAT, 'semfeat_sparse_10_c_10_sknn_80_neg_300.train'), 10)
# dump_semfeat(x_tt, os.path.join(DIR_FEAT, 'semfeat_sparse_10_c_10_sknn_80_neg_300.test'), 10)

# ## ------------------
# fn_test = '/home/phong/imdbs/VOC07/groups_367_sparse_10/semfeat_img_test_sparse_10_c_10_sknn_80_neg_100k.fake.libsvm'
# fn_train = '/home/phong/imdbs/VOC07/groups_367_sparse_10/semfeat_img_train_sparse_10_c_10_sknn_80_neg_100k.fake.libsvm'
# DIR_FEAT = '/home/phong/imdbs/VOC07/groups_367_sparse_10'
	
# x_tn = read_semfeat(fn_train)
# x_tt = read_semfeat(fn_test)

# dump_semfeat(x_tn, os.path.join(DIR_FEAT, 'semfeat_sparse_10_c_10_sknn_80_neg_100k.train'), 10)
# dump_semfeat(x_tt, os.path.join(DIR_FEAT, 'semfeat_sparse_10_c_10_sknn_80_neg_100k.test'), 10)

##
# fn_test = '/home/phong/imdbs/VOC07/groups_367_sparse_5/semfeat_sparse_5_c_10_sknn_80_neg_300.fake.test'
# fn_train = '/home/phong/imdbs/VOC07/groups_367_sparse_5/semfeat_sparse_5_c_10_sknn_80_neg_300.fake.train'
# DIR_FEAT = '/home/phong/imdbs/VOC07/groups_367_sparse_5'

# x_tn = read_semfeat(fn_train)
# x_tt = read_semfeat(fn_test)

# dump_semfeat(x_tn, os.path.join(DIR_FEAT, 'semfeat_sparse_5_c_10_sknn_80_neg_300.train'), 5)
# dump_semfeat(x_tt, os.path.join(DIR_FEAT, 'semfeat_sparse_5_c_10_sknn_80_neg_300.test'), 5)

# ##
# fn_test = '/home/phong/imdbs/VOC07/groups_367_sparse_5/semfeat_sparse_5_c_10_sknn_80_neg_100k.fake.test'
# fn_train = '/home/phong/imdbs/VOC07/groups_367_sparse_5/semfeat_sparse_5_c_10_sknn_80_neg_100k.fake.train'
# DIR_FEAT = '/home/phong/imdbs/VOC07/groups_367_sparse_5'

# x_tn = read_semfeat(fn_train)
# x_tt = read_semfeat(fn_test)

# dump_semfeat(x_tn, os.path.join(DIR_FEAT, 'semfeat_sparse_5_c_10_sknn_80_neg_100k.train'), 5)
# dump_semfeat(x_tt, os.path.join(DIR_FEAT, 'semfeat_sparse_5_c_10_sknn_80_neg_100k.test'), 5)
