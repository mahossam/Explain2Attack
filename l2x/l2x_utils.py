import pandas as pd 
import numpy as np 
try:
	import cPickle as pickle
except:
	import pickle
import os 
import csv
from bs4.dammit import EntitySubstitution

def get_selected_words(x_single, score, id_to_word, k): 
	selected_words = {} # {location: word_id}

	selected = np.argsort(score)[-k:] 
	selected_k_hot = np.zeros(x_single.shape[0])  # number of words per sentence
	selected_k_hot[selected] = 1.0

	x_selected = (x_single * selected_k_hot).astype(int)
	return x_selected 

def create_dataset_from_score(x, name, scores, k, args):
	with open(os.path.join(args.outdir, 'id_to_word.pkl'),'rb') as f:
		id_to_word = pickle.load(f)
	new_data = []
	new_texts = []
	for i, x_single in enumerate(x):
		x_selected = get_selected_words(x_single, 
			scores[i], id_to_word, k)

		new_data.append(x_selected) 

	np.save(os.path.join(args.outdir, name+'-L2X.npy'), np.array(new_data))

def calculate_acc(pred, y):
	return np.mean(np.argmax(pred, axis = 1) == np.argmax(y, axis = 1))


def save_sent_viz_file(x, name, scores, k, args):
	escaper = EntitySubstitution()

	with open(os.path.join(args.outdir, 'id_to_word.pkl'),'rb') as f:
		id_to_word = pickle.load(f)
	new_data = list()
	new_texts = list()
	with open(os.path.join(args.outdir, 'sent_viz_L2X'+name+'.html'), 'w') as txt_file:
		txt_file.write(u"<!DOCTYPE html>\n<html>\n<body>\n".encode("utf-8"))

		for i, x_single in enumerate(x):
			x_selected = get_selected_words(x_single,
				scores[i], id_to_word, k)

			# new_data.append(x_selected)
			for s_i, s in enumerate(x_single):
				if s != 0:
					break

			# txt_file.write( (u" ".join([id_to_word[i] for i in x_single[s_i:] if i != 0]) + u"\n").encode("utf-8") )

			sent_viz = list()
			for wp, wi in enumerate(x_single[s_i:]):
				# if x_selected[s_i:][wp] != 0:
				# 	placeholder = u"-" * len(id_to_word[wi])
				# else:
				# 	placeholder = u" " * len(id_to_word[wi])
				if x_selected[s_i:][wp] != 0:
					placeholder = u"<mark><strong>" + escaper.substitute_html(id_to_word[wi]) + u"</strong></mark>"
				else:
					placeholder = escaper.substitute_html(id_to_word[wi])

				sent_viz.append(placeholder)

			txt_file.write((u"<p>" + u" ".join(sent_viz) + u"</p><br>\n").encode("utf-8"))
		txt_file.write(u"</body>\n</html>\n".encode("utf-8"))


def export_to_Jin_paper_format_bert(fname, x, y, id_to_word):
	textfooler_id_to_word = id_to_word.copy()
	textfooler_id_to_word[[key for (key, value) in id_to_word.items() if value == "<UNK>"][0]] = "[UNK]"
	textfooler_id_to_word[[key for (key, value) in id_to_word.items() if value == "<PAD>"][0]] = "[PAD]"
	with open(fname, 'w') as data_file:
		for idx, (label, text) in enumerate(zip(y, x)):
			data_file.write(
				(unicode(str(label)) + u" " + u" ".join([textfooler_id_to_word[wi] for wi in text]) + u"\n").encode("utf-8"))


def str_to_bool(s):
	if s.lower() == "yes" or s.lower() == "true":
		return True
	else:
		return False

