"""
David R. Winer
drwiner@cs.utah.edu
Machine Learning HW1 Implementation
"""

from collections import namedtuple
import math

Label = namedtuple('Label', ['label', 'firstname', 'middlename', 'lastname'])


#### FEATURES ####


# Feature 1
def firstname_longer_lastname(lb):
	if len(lb.firstname) > len(lb.lastname):
		return True
	return False


# Feature 2
def has_middle_name(lb):
	if lb.middlename is not None:
		return True
	return False


# Feature 3
def same_first_and_last_letter(lb):
	if lb.firstname[0].lower() == lb.firstname[-1]:
		return True
	return False


# Feature 4
def firstnameletter_less_lastnameletter(lb):
	if lb.firstname[0].lower() < lb.lastname[0].lower():
		return True
	return False


# Feature 5
def firstnameletter_is_vowel(lb):
	if lb.firstname[0].lower() in {'a', 'e', 'i', 'o', 'u'}:
		return True
	return False


# Feature 6
def even_length_lastname(lb):
	if len(lb.lastname) % 2 == 0:
		return True
	return False


# Set of Features, initially
FEATURES = [firstname_longer_lastname,
            has_middle_name,
            same_first_and_last_letter,
            firstnameletter_less_lastnameletter,
            firstnameletter_is_vowel,
            even_length_lastname]


#### EQUATIONS ####


def gain(samples, feature):
	pos_samples = [lbl for lbl in samples if feature(lbl)]
	neg_samples = [lbl for lbl in samples if not feature(lbl)]
	pos_gain = (len(pos_samples) / len(samples)) * entropy(pos_samples)
	neg_gain = (len(neg_samples) / len(samples)) * entropy(neg_samples)
	return entropy(samples) - (pos_gain + neg_gain)


def entropy(samples):
	total = len(samples)
	sum_pos = sum(1 for lbl in samples if lbl.label == '+')
	sum_neg = sum(1 for lbl in samples if lbl.label == '-')
	p_pos = sum_pos / total
	p_neg = sum_neg / total
	return -p_pos * math.log2(p_pos) - p_neg * math.log2(p_neg)


#### ID3 and DECISION TREE ####

"""
Nodes are nested dictionaries with 3 values, a feature, a 0 : {}, and a 1 : {}
"""

def num_samples_with_label(samples, target_label):
	return sum(1 for lbl in samples if lbl.label == target_label)


def all_samples_target(samples, target_label):
	return len(samples) == num_samples_with_label(samples, target_label)


def best_feature(samples, features):
	best = (100000, None)
	for feature in features:
		x = gain(samples, feature)
		if x < best[0]:
			best = (x, feature)
	return best[1]


def ID3(samples, features):

	if len(features) == 0:
		# return most common value of remaining samples
		if num_samples_with_label(samples, '+') > num_samples_with_label(samples, '-'):
			return {'feature':None, 1: True, 0: False}
		else:
			return {'feature': None, 0: True, 1: False}

	# Pick Best Feature
	best_f = best_feature(samples, features)

	# if all samples have positive label
	if all_samples_target(samples, '+'):
		return {'feature': best_f, 0: False, 1: True}

	# if all samples have negative label
	if all_samples_target(samples, '-'):
		return {'feature': best_f, 1: False, 0: True}

	# feature is True, S_{v=1}
	sub_class_pos = {lbl for lbl in samples if best_f(lbl) is True}
	if len(sub_class_pos) == len(samples):
		pos_child = True
	# elif there's no samples, then choose most common label as generalization
	elif len(sub_class_pos) == 0:
		if num_samples_with_label(samples, '+') > num_samples_with_label(samples, '-'):
			pos_child = True
		else:
			pos_child = False
	else:
		pos_child = ID3(sub_class_pos, set(features) - set(best_f))

	# feature is False
	sub_class_neg = {lbl for lbl in samples if best_f(lbl) is False}
	if len(sub_class_neg) == len(samples):
		neg_child = True
	# elif there's no samples, then choose most common label as generalization
	elif len(sub_class_neg) == 0:
		if num_samples_with_label(samples, '+') > num_samples_with_label(samples, '-'):
			neg_child = False
		else:
			neg_child = True
	else:
		neg_child = ID3(sub_class_neg, set(features) - set(best_f))

	return {'feature': best_f, 1: pos_child, 0: neg_child}


def use_tree(tree, item):

	# base case, the tree is a value
	if type(tree) is bool:
		return tree

	# otherwise, recursively evaluate item with features
	result = tree['feature'](item)
	if result:
		return use_tree(tree[1], item)
	else:
		return use_tree(tree[0], item)


if __name__ == '__main__':
	training_data = []
	with open('data//training.data', 'r') as training_data_file:
		for line in training_data_file:
			sp = line.split()
			if len(sp) > 3:
				# has middle name
				lb = Label(sp[0], sp[1], sp[2], sp[3])
			else:
				lb = Label(sp[0], sp[1], None, sp[2])
			training_data.append(lb)

	dtree = ID3(FEATURES, training_data)




			# implement decision tree
			# implement ID3 algorithm
			# perform cross validation experiment and report error
			# Limit depth and repeat
