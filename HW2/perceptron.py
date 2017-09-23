# Run on Lab1-19 to access numpy module

# David R. Winer
# drwiner at cs.utah.edu

from collections import namedtuple, defaultdict
import numpy as np

# LabeledEx = namedtuple('LabeledEx', ['label', 'feature_dict'])
LabeledEx = namedtuple('LabeledEx', ['label', 'feature_vec'])

#### HELPER METHODS ####

# vector is string of form <index:value, index:value,....>; indices are "int", values are "float"
def vector_to_dict(vector):
	item_dict = defaultdict(int)
	for item in vector:
		index, value = item.split(':')
		item_dict[int(index)] = float(value)
	return item_dict


def vector_to_list(vector):
	feat_dict = vector_to_dict(vector)
	local_max = max(feat_dict, key=int)
	precursor_list = [feat_dict[j] for j in range(local_max)]
	return precursor_list


def dot_weight_dict(w, d):
	dot_prod = 0
	for j, weight in enumerate(w):
		dot_prod += weight*d[j]
	return dot_prod


def dot_dict(d1, d2, n):
	dot_prod = 0
	for i in range(1,n+1):
		dot_prod += d1[i]*d2[i]
	return dot_prod


# extract labeled examples from data file
def get_data(data_file_name):
	data = []
	with open(data_file_name, 'r') as training_data_file:
		for line in training_data_file:
			example = line.split()
			# arg1 is label, arg2 is dictionary (keys are indices, values are values)
			example_tup = LabeledEx(int(example[0]), vector_to_list(example[1:]))
			data.append(example_tup)

	# now, find the maximum size vector and pad all others
	n = get_longest_vec(data)
	training_data = []
	for example in data:
		precursor_list = example.feat_vec
		if len(example.feat_vec) < n:
			precursor_list.expand([0 for j in range(n-len(example.feat_vec))])
			if len(precursor_list) != n:
				raise ArithmeticError('bad arithmetic')
		training_data.append(LabeledEx(example[0], np.array(precursor_list)))

	return training_data


def get_longest_vec(examples):
	return max(len(example.feat_vec) for example in examples)


def get_max_index(examples):
	max_index = 0
	for ex in examples:
		feat_dict = ex.feature_dict
		local_max = max(feat_dict, key=int)
		if local_max > max_index:
			max_index = local_max
	return max_index


#### PERCEPTRON METHODS ####


def update_weights(w, lr, label, feat_vec):
	update_op = feat_vec*lr

	# mistake on positive
	if label == 1:
		new_w = w + update_op

	# mistake on negative
	else:
		new_w = w - update_op

	return new_w


def simple_perceptron_wrapper(examples, epochs):
	learning_rates = [1, 0.1, 0.01]

	# examples are precompiled to have correct length
	num_feats = len(examples[0].feat_vec)

	# initialize weights
	weights = np.array([0 for j in range(num_feats)])

	# run perceptron for each learning rate
	w_values = [simple_perceptron(examples, weights, lr, epochs) for lr in learning_rates]

	return w_values


def simple_perceptron(examples, weights, learning_rate, epochs):

	for epoch in range(epochs):
		# weights = before each epoch; initialize w_prime as weights before each epoch
		w_prime = weights
		for example in examples:
			y_prime = np.dot(weights, example.feat_vec)
			if y_prime >= 0:
				y_prime = 1
			else:
				y_prime = 0
			if y_prime != examples.label:
				# update weights
				w_prime = update_weights(w_prime, learning_rate, examples.label, examples.feat_vec)

		# BATCH : update weights only after each epoch
		weights = w_prime
	return weights


def simple_perceptron_test():
	pass


def cross_validate(cv_split, perceptron_method, pc_test, epochs):
	results = []

	# each 'i' is test
	for i in range(4):
		# each 'j' is training
		training = []
		for j in range(4):
			if i == j:
				continue
			training.extend(cv_split[j])

		# train with 4/5
		weight_vals = perceptron_method(training, epochs)

		# test on i
		result = pc_test(cv_split[i], weight_vals)
		results.append(result)

	# avg over results
	pass



if __name__ == '__main__':
	training_dev = get_data('DataSet//phishing.dev')

	training_cross_val = []
	for i in range(4):
		fold = get_data('DataSet//CVSplits//training0{}.data'.format(str(i)))
		training_cross_val.append(fold)

	# 1. Simple Perceptron
	cross_validate(training_cross_val, simple_perceptron_wrapper, simple_perceptron_test, 10)
