def ID3_OLD(samples, features, target_labels):
	# if all samples have positive label

	for tlabel in target_labels:
		if all_samples_target(samples, tlabel):
			return tlabel

	if len(features) == 0:
		# return most common value of remaining samples
		return most_labeled(samples, target_labels)

	# Pick Best Feature
	if len(features) == 1:
		best_f = list(features)[0]
	else:
		best_f = best_feature(samples, features)

	# feature is True, S_{v=1}
	sub_class_pos = {lbl for lbl in samples if best_f(lbl) is True}
	if len(sub_class_pos) == len(samples):
		pos_child = True
	# elif there's no samples, then choose most common label as generalization
	elif len(sub_class_pos) == 0:
		if num_samples_with_label(samples, True) > num_samples_with_label(samples, False):
			pos_child = True
		else:
			pos_child = False
	else:
		pos_child = ID3_OLD(sub_class_pos, set(features) - {best_f},target_labels)

	# feature is False
	sub_class_neg = {lbl for lbl in samples if best_f(lbl) is False}
	if len(sub_class_neg) == len(samples):
		neg_child = True
	# elif there's no samples, then choose most common label as generalization
	elif len(sub_class_neg) == 0:
		if num_samples_with_label(samples, True) > num_samples_with_label(samples, False):
			neg_child = True
		else:
			neg_child = False
	else:
		neg_child = ID3_OLD(sub_class_neg, set(features) - {best_f}, target_labels)

	return {'feature': best_f, 1: pos_child, 0: neg_child}