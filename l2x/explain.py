"""
The code for constructing the original word-CNN is based on
https://github.com/keras-team/keras/blob/master/examples/imdb_cnn.py
"""
from __future__ import absolute_import, division, print_function   
from keras.layers import Conv1D, Input, GlobalMaxPooling1D, Multiply, Lambda, Embedding, Dense, Dropout, Activation
from keras.datasets import imdb
from keras.engine.topology import Layer 
from keras import backend as K  
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.callbacks import ModelCheckpoint
from keras.models import Model, Sequential
from keras.backend.tensorflow_backend import set_session

import numpy as np
import tensorflow as tf
import time 
import numpy as np 
import sys
import os
# import urllib2
import urllib
import tarfile
import zipfile
try:
	import cPickle as pickle
except:
	import pickle
import os 
from l2x_utils import create_dataset_from_score, calculate_acc, save_sent_viz_file, export_to_Jin_paper_format_bert, str_to_bool
import l2x_dataloader
from l2x_dataloader import read_corpus, load_embedding
from bert.l2x_Dataset_BERT import Dataset_BERT, convert_examples_to_features

# Set parameters:
# rseeds=[10086, 10087, 10088, 10089, 10090]
max_features = -1  # 29528  #5000
maxlen = -1  # 256 # 400
batch_size = -1  # 64  # 40
embedding_dims = 50  # 100  # 50
# embedding_dims = 300
filters = 250
kernel_size = 3
hidden_dims = 250
# hidden_dims = 250
epochs = 5
k = -1  # 150  #20  # Number of selected words by L2X.
PART_SIZE = 125

###########################################
###############Load data###################
###########################################
label_map = {"imdb": [0, 1], "yelp": [1, 2], "fake": [1, 2], "ag": [1, 2, 3, 4], "mr": [0, 1], "amazon": [0, 1], "amazon_movies_small": [0, 1], "amazon_movies_20K": [0, 1]}
os.environ['TF_CUDNN_DETERMINISM'] = '1'

def post_process_trim_then_pad_right(sequences, pad_id, trim_length, max_length):
	ret = [seq[:trim_length] + [pad_id] * (max_length - trim_length) for seq in sequences]
	return ret

def load_data_bert():
	"""
	Load data if data have been created.
	Create data otherwise.

	"""
	global max_features
	if 'data' not in os.listdir('.'):
		os.mkdir('data')

	# if 'id_to_word.pkl' not in os.listdir('data'):
	print('Loading data...')

	# texts_train, labels_train = read_corpus(os.path.join(os.path.dirname(args.test_data_path), 'train_tok.csv'), clean=False)
	texts_train, labels_train = read_corpus(args.train_data_path, clean=False, max_lines=args.source_data_size)
	texts_test, labels_test = read_corpus(args.test_data_path, clean=str_to_bool(args.clean_test_data), max_lines=args.source_data_size)
	if args.target_data_path != "":
		texts_explain, _ = read_corpus(args.target_data_path, clean=str_to_bool(args.clean_test_data))

	ntrain, ntest = len(labels_train), len(labels_test)
	texts_train, labels_train = texts_train[:(ntrain - ntrain % batch_size)], labels_train[:(ntrain - ntrain % batch_size)]
	texts_test, labels_test = texts_test[:(ntest - ntest % batch_size)], labels_test[:(ntest - ntest % batch_size)]

	dataset = Dataset_BERT(args.target_model_path, batch_size=batch_size)
	word_to_id = dict(dataset.tokenizer.vocab)
	id_to_word = {value: key for key, value in word_to_id.items()}

	train_label_list = label_map[args.dataset_name]
	test_label_list = train_label_list
	if "author" in args.test_data_path:
		print("Special case of input test file, all labels are already zero based")
		test_label_list = [l for l in xrange(len(list(set(labels_test))))]
	print(test_label_list)
	x_val, _y_val = convert_examples_to_features(texts_test, args.source_max_seq_length, dataset.tokenizer, labels_test, label_list=test_label_list, bert_tokenize=str_to_bool(args.bert_tokenize))
	x_train, _y_train = convert_examples_to_features(texts_train, args.source_max_seq_length, dataset.tokenizer, labels_train, label_list=train_label_list, bert_tokenize=str_to_bool(args.bert_tokenize))

	print("Test labels = ", list(set(_y_val)))
	print("Train labels = ", list(set(_y_train)))

	x_train, x_val = np.array(x_train), np.array(x_val)
	max_features = len(id_to_word.keys())
	print("Data import finished!")

	print(len(x_train), 'train sequences')
	print(len(x_val), 'test sequences')

	x_explain = None
	if args.target_data_path != "":
		# TODO, Investigate this, how to make sure BERT datalengths match across L2 training and attack set explaining (-100 is a dummy instead of args.target_max_seq_length til this is resolved)?
		x_explain, _ = convert_examples_to_features(texts_explain, args.target_max_seq_length, dataset.tokenizer, None, label_list=None, bert_tokenize=str_to_bool(args.bert_tokenize))
		x_explain = post_process_trim_then_pad_right(x_explain, 0, args.target_max_seq_length, args.source_max_seq_length)
		x_explain = np.array(x_explain)
		print(len(x_explain), 'explain sequences')

	y_train = np.eye(4)[_y_train] if args.dataset_name == "ag" else np.eye(2)[_y_train]
	y_val = np.eye(4)[_y_val] if args.dataset_name == "ag" else np.eye(2)[_y_val]

	# np.save('./data/x_train.npy', x_train)
	# np.save('./data/y_train.npy', y_train)
	# np.save('./data/x_val.npy', x_val)
	# np.save('./data/y_val.npy', y_val)
	with open(os.path.join(args.outdir, 'id_to_word.pkl'), 'wb') as f:
		pickle.dump(id_to_word, f, protocol=0)

	# else:
	# 	x_train, y_train, x_val, y_val = np.load('data/x_train.npy', allow_pickle=True),np.load('data/y_train.npy', allow_pickle=True),np.load('data/x_val.npy', allow_pickle=True),np.load('data/y_val.npy', allow_pickle=True)
	# 	with open('data/id_to_word.pkl','rb') as f:
	# 		id_to_word = pickle.load(f)
	# 		max_features = len(id_to_word.keys())

	return x_train, y_train, x_val, y_val, id_to_word, x_explain

def load_data_bert_explain():
	"""
	Load data if data have been created.
	Create data otherwise.

	"""
	global max_features
	if args.target_data_path != "":
		texts_explain, _ = read_corpus(args.target_data_path, clean=str_to_bool(args.clean_test_data))

	dataset = Dataset_BERT(args.target_model_path, batch_size=batch_size)
	word_to_id = dict(dataset.tokenizer.vocab)
	id_to_word = {value: key for key, value in word_to_id.items()}

	max_features = len(id_to_word.keys())

	x_explain = None
	if args.target_data_path != "":
		x_explain, _ = convert_examples_to_features(texts_explain, args.target_max_seq_length, dataset.tokenizer, None,
													label_list=None, bert_tokenize=str_to_bool(args.bert_tokenize))
		x_explain = post_process_trim_then_pad_right(x_explain, 0, args.target_max_seq_length,
													 args.source_max_seq_length)
		x_explain = np.array(x_explain)
		# print(len(x_explain), 'explain sequences')

	# with open(os.path.join(args.outdir, 'id_to_word.pkl'), 'wb') as f:
	# 	pickle.dump(id_to_word, f, protocol=0)

	return id_to_word, x_explain


def load_data_CNN_LSTM():
	"""
	Load data if data have been created.
	Create data otherwise.
	"""
	global max_features
	if 'data' not in os.listdir('.'):
		os.mkdir('data')

	# if 'id_to_word.pkl' not in os.listdir('data'):
	print('Loading data (CNN/LSTM target) ...')

	fix_labels = False
	if args.dataset_name == "yelp" or args.dataset_name == "fake" or args.dataset_name == "ag":
		fix_labels = True

	test_fix_labels = fix_labels
	if "author" in args.test_data_path:
		print("Special case of input test file, all labels are already zero based")
		test_fix_labels = False

	# texts_train, labels_train = read_corpus(os.path.join(os.path.dirname(args.test_data_path), 'train_tok.csv'), clean=False)
	texts_train, labels_train = read_corpus(args.train_data_path, clean=False, max_lines=args.source_data_size, fix_labels=fix_labels, max_length=args.source_max_seq_length)
	texts_test, labels_test = read_corpus(args.test_data_path, clean=str_to_bool(args.clean_test_data), max_lines=args.source_data_size, fix_labels=test_fix_labels, max_length=args.source_max_seq_length)
	if args.target_data_path != "":
		texts_explain, _ = read_corpus(args.target_data_path, clean=str_to_bool(args.clean_test_data), fix_labels=fix_labels, max_length=args.target_max_seq_length)

	ntrain, ntest = len(labels_train), len(labels_test)
	texts_train, labels_train = texts_train[:(ntrain - ntrain % batch_size)], labels_train[:(ntrain - ntrain % batch_size)]
	texts_test, labels_test = texts_test[:(ntest - ntest % batch_size)], labels_test[:(ntest - ntest % batch_size)]

	embs = l2x_dataloader.load_embedding(args.word_embeddings_path)
	word2id = dict()
	if embs is not None:
		embwords = embs
		for word in embwords:
			assert word not in word2id, "Duplicate words in pre-trained embeddings"
			word2id[word] = len(word2id)

		sys.stdout.write("{} pre-trained word embeddings loaded.\n".format(len(word2id)))
		# n_d = len(embvecs[0])

	oov, pad = '<oov>', '<pad>'
	if oov not in word2id:
		word2id[oov] = len(word2id)

	if pad not in word2id:
		word2id[pad] = len(word2id)

	n_V = len(word2id)  # , n_d
	oovid = word2id[oov]
	padid = word2id[pad]
	# self.embedding = nn.Embedding(self.n_V, n_d)
	# self.embedding.weight.data.uniform_(-0.25, 0.25)

	word_to_id = word2id
	id_to_word = {value: key for key, value in word_to_id.items()}
	x_train, _y_train = l2x_dataloader.create_batches(texts_train, labels_train, map2id=word_to_id, max_len=args.source_max_seq_length)
	x_val, _y_val = l2x_dataloader.create_batches(texts_test, labels_test, map2id=word_to_id, max_len=args.source_max_seq_length)

	ulbl_train = list(set(_y_train))
	ulbl_val = list(set(_y_val))
	if 0 not in ulbl_train + ulbl_val:
		print("WARNING: 0 not found in labels. Make sure that labels are zero-based")

	print("Test labels = ", ulbl_val)
	print("Train labels = ", ulbl_train)

	x_train, x_val = np.array(x_train), np.array(x_val)
	max_features = len(id_to_word.keys())
	print("Data import finished!")

	print(len(x_train), 'train sequences')
	print(len(x_val), 'test sequences')

	x_explain = None
	if args.target_data_path != "":
		x_explain = l2x_dataloader.create_batches_x(texts_explain, map2id=word_to_id, max_len=args.target_max_seq_length)
		x_explain = post_process_trim_then_pad_right(x_explain, padid, args.target_max_seq_length, args.source_max_seq_length)
		x_explain = np.array(x_explain)
		print(len(x_explain), 'explain sequences')

	y_train = np.eye(4)[_y_train] if args.dataset_name == "ag" else np.eye(2)[_y_train]
	y_val = np.eye(4)[_y_val] if args.dataset_name == "ag" else np.eye(2)[_y_val]

	# np.save('./data/x_train.npy', x_train)
	# np.save('./data/y_train.npy', y_train)
	# np.save('./data/x_val.npy', x_val)
	# np.save('./data/y_val.npy', y_val)
	with open(os.path.join(args.outdir, 'id_to_word.pkl'), 'wb') as f:
		pickle.dump(id_to_word, f, protocol=0)

	# else:
	# 	x_train, y_train, x_val, y_val = np.load('data/x_train.npy', allow_pickle=True),np.load('data/y_train.npy', allow_pickle=True),np.load('data/x_val.npy', allow_pickle=True),np.load('data/y_val.npy', allow_pickle=True)
	# 	with open('data/id_to_word.pkl','rb') as f:
	# 		id_to_word = pickle.load(f)
	# 		max_features = len(id_to_word.keys())

	return x_train, y_train, x_val, y_val, id_to_word, x_explain


# def old_load_data_imdb():
# 	# # save np.load
# 	np_load_old = np.load
# 	# # modify the default parameters of np.load
# 	np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
# 	max_features = 5000
# 	(x_train, _y_train), (x_val, _y_val) = imdb.load_data(num_words=max_features, index_from=3)
# 	np.load = np_load_old
# 	word_to_id = imdb.get_word_index()
# 	word_to_id = {k: (v + 3) for k, v in word_to_id.items()}
# 	word_to_id["<PAD>"] = 0
# 	word_to_id["<START>"] = 1
# 	word_to_id["<UNK>"] = 2
# 	id_to_word = {value: key for key, value in word_to_id.items()}
#
# 	print('Pad sequences (samples x time)')
# 	x_train = sequence.pad_sequences(x_train, maxlen=maxlen + 1, padding='post',
# 									 truncating='post')  # maxlen includes a start character (can be removed if desired)
# 	x_val = sequence.pad_sequences(x_val, maxlen=maxlen + 1, padding='post', truncating='post')
# 	x_train = x_train[:, 1:]  # Remove the <start> character
# 	x_val = x_val[:, 1:]
#
# 	# export_to_Jin_paper_format_bert('imdb_v'+str(max_features)+'_full_train.txt', x_train, _y_train, id_to_word)
# 	# export_to_Jin_paper_format_bert('imdb_v'+str(max_features)+'_full_test.txt', x_val, _y_val, id_to_word)
#
# 	return _y_train, _y_val, id_to_word, x_train, x_val


###########################################
###############Original Model##############
###########################################

def create_original_model():
	"""
	Build the original model to be explained. 

	"""
	model = Sequential()
	model.add(Embedding(max_features,
						embedding_dims,
						input_length=maxlen))
	model.add(Dropout(0.2))
	model.add(Conv1D(filters,
					 kernel_size,
					 padding='valid',
					 activation='relu',
					 strides=1))
	model.add(GlobalMaxPooling1D())
	model.add(Dense(hidden_dims))
	model.add(Dropout(0.2))
	model.add(Activation('relu'))
	model.add(Dense(2))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy',
				  optimizer='adam',
				  metrics=['accuracy'])

	return model


def generate_original_preds(train = True): 
	"""
	Generate the predictions of the original model on training
	and validation datasets. 

	The original model is also trained if train = True. 

	"""
	x_train, y_train, x_val, y_val, id_to_word, _ = load_data_bert()
	model = create_original_model()

	if train:
		filepath="models/original.hdf5"
		checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 
			verbose=1, save_best_only=True, mode='max')
		callbacks_list = [checkpoint]
		model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks = callbacks_list, epochs=epochs, batch_size=batch_size)

	model.load_weights('./models/original.hdf5', 
		by_name=True) 

	pred_train = model.predict(x_train,verbose = 1, batch_size = 1000)
	pred_val = model.predict(x_val,verbose = 1, batch_size = 1000)
	if not train:
		print('The val accuracy is {}'.format(calculate_acc(pred_val,y_val)))
		print('The train accuracy is {}'.format(calculate_acc(pred_train,y_train)))


	np.save(os.path.join(args.outdir, 'pred_train.npy'), pred_train)
	np.save(os.path.join(args.outdir, 'pred_val.npy'), pred_val)

	np.save(os.path.join(args.outdir, 'labels_model_train.npy'), np.argmax(pred_train, axis=-1))
	np.save(os.path.join(args.outdir, 'labels_model_val.npy'), np.argmax(pred_val, axis=-1))

###########################################
####################L2X####################
###########################################
# Define various Keras layers.
Mean = Lambda(lambda x: K.sum(x, axis = 1) / float(k), 
	output_shape=lambda x: [x[0],x[2]]) 


class Concatenate(Layer):
	"""
	Layer for concatenation. 
	
	"""
	def __init__(self, **kwargs): 
		super(Concatenate, self).__init__(**kwargs)

	def call(self, inputs):
		input1, input2 = inputs  
		input1 = tf.expand_dims(input1, axis = -2) # [batchsize, 1, input1_dim] 
		dim1 = int(input2.get_shape()[1])
		input1 = tf.tile(input1, [1, dim1, 1])
		return tf.concat([input1, input2], axis = -1)

	def compute_output_shape(self, input_shapes):
		input_shape1, input_shape2 = input_shapes
		input_shape = list(input_shape2)
		input_shape[-1] = int(input_shape[-1]) + int(input_shape1[-1])
		input_shape[-2] = int(input_shape[-2])
		return tuple(input_shape)


class TileWrapper(Layer):
	"""
	Layer for concatenation.

	"""

	def __init__(self, max_len, **kwargs):
		super(TileWrapper, self).__init__(**kwargs)
		self.max_len = max_len

	def call(self, input):
		_input = tf.expand_dims(input, axis=-2)  # [batchsize, 1, input1_dim]
		dim1 = self.max_len
		_input = tf.tile(_input, [1, dim1, 1])
		return _input

class Sample_Concrete(Layer):
	"""
	Layer for sample Concrete / Gumbel-Softmax variables. 

	"""
	def __init__(self, tau0, k, **kwargs): 
		self.tau0 = tau0
		self.k = k
		super(Sample_Concrete, self).__init__(**kwargs)

	def call(self, logits):   
		# logits: [batch_size, d, 1]
		logits_ = K.permute_dimensions(logits, (0,2,1))# [batch_size, 1, d]

		d = int(logits_.get_shape()[2])
		unif_shape = [batch_size,self.k,d]

		uniform = K.random_uniform_variable(shape=unif_shape,
			low = np.finfo(tf.float32.as_numpy_dtype).tiny,
			high = 1.0, seed=args.seed)
		gumbel = - K.log(-K.log(uniform))
		noisy_logits = (gumbel + logits_)/self.tau0
		samples = K.softmax(noisy_logits)
		samples = K.max(samples, axis = 1) 
		logits = tf.reshape(logits,[-1, d]) 
		threshold = tf.expand_dims(tf.nn.top_k(logits, self.k, sorted = True)[0][:,-1], -1)
		discrete_logits = tf.cast(tf.greater_equal(logits,threshold),tf.float32)
		
		output = K.in_train_phase(samples, discrete_logits)
		return tf.expand_dims(output,-1)

	def compute_output_shape(self, input_shape):
		return input_shape


def construct_gumbel_selector(X_ph, num_words, embedding_dims, maxlen):
	"""
	Build the L2X model for selecting words. 

	"""
	emb_layer = Embedding(num_words, embedding_dims, input_length = maxlen, name = 'emb_gumbel')
	emb = emb_layer(X_ph) #(400, 50) 
	net = Dropout(0.2, name = 'dropout_gumbel')(emb)
	net = emb
	first_layer = Conv1D(100, kernel_size, padding='same', activation='relu', strides=1, name = 'conv1_gumbel')(net)    

	# global info
	net_new = GlobalMaxPooling1D(name = 'new_global_max_pooling1d_1')(first_layer)
	global_info = Dense(100, name = 'new_dense_1', activation='relu')(net_new)
	# combined = TileWrapper(maxlen)(global_info)

	# local info
	net = Conv1D(100, 3, padding='same', activation='relu', strides=1, name = 'conv2_gumbel')(first_layer)
	local_info = Conv1D(100, 3, padding='same', activation='relu', strides=1, name = 'conv3_gumbel')(net)
	combined = Concatenate()([global_info,local_info])
	# combined = local_info

	net = Dropout(0.2, name = 'new_dropout_2')(combined)
	net = Conv1D(100, 1, padding='same', activation='relu', strides=1, name = 'conv_last_gumbel')(net)   

	logits_T = Conv1D(1, 1, padding='same', activation=None, strides=1, name = 'conv4_gumbel')(net)  
	
	return logits_T


def L2X(train=True):
	"""
	Generate scores on features on validation by L2X.

	Train the L2X model with variational approaches 
	if train = True. 

	"""
	# print('Loading dataset...')
	if train:
		load_data = load_data_bert if args.target_model == "bert" else load_data_CNN_LSTM
		if args.target_model == "wordCNN" or args.target_model == "wordLSTM":
			print("WARNING: intended to attack CNN or LSTM models. Make sure that source/target_max_seq_length match target model training lengths")
		x_train, y_train, x_val, y_val, id_to_word, x_explain = load_data()
		pred_train = np.load(args.train_pred_labels, allow_pickle=True)
		pred_val = np.load(args.val_pred_labels, allow_pickle=True)
		x_train, y_train, x_val, y_val = x_train[:pred_train.shape[0]], y_train[:pred_train.shape[0]], x_val[:pred_train.shape[0]], y_val[:pred_train.shape[0]]
	else:
		load_data_explain = load_data_bert_explain if args.target_model == "bert" else None
		id_to_word, x_explain = load_data_explain()
	# print('Creating model...')

	# P(S|X)
	with tf.variable_scope('selection_model'):
		X_ph = Input(shape=(maxlen,), dtype='int32')

		logits_T = construct_gumbel_selector(X_ph, max_features, embedding_dims, maxlen)
		tau = 0.5 
		T = Sample_Concrete(tau, k)(logits_T)

	# q(X_S)
	with tf.variable_scope('prediction_model'):
		emb2 = Embedding(max_features, embedding_dims, 
			input_length=maxlen)(X_ph)

		net = Mean(Multiply()([emb2, T]))
		net = Dense(hidden_dims)(net)
		net = Activation('relu')(net) 
		preds = Dense((4 if args.dataset_name == "ag" else 2), activation='softmax',
			name = 'new_dense')(net)

	model = Model(inputs=X_ph, outputs=preds)
	model.compile(loss='categorical_crossentropy',
				  optimizer='rmsprop',#optimizer,
				  metrics=['acc']) 
	# train_acc = np.mean(np.argmax(pred_train, axis = 1)==np.argmax(y_train, axis = 1))
	# val_acc = np.mean(np.argmax(pred_val, axis = 1)==np.argmax(y_val, axis = 1))
	# print('The train and validation accuracy of the original model is {} and {}'.format(train_acc, val_acc))

	if train:
		filepath = os.path.join(args.outdir, "l2x.hdf5")
		checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 
			verbose=1, save_best_only=True, mode='max')
		callbacks_list = [checkpoint] 
		st = time.time()
		model.fit(x_train, pred_train,
			validation_data=(x_val, pred_val),
			callbacks = callbacks_list,
			epochs=5, batch_size=batch_size, verbose=0)
		duration = time.time() - st
		print('Training time is {}'.format(duration))

	model.load_weights(os.path.join(args.outdir, 'l2x.hdf5'), by_name=True)

	pred_model = Model(X_ph, logits_T)
	pred_model.compile(loss='categorical_crossentropy', 
		optimizer='adam', metrics=['acc']) 

	st = time.time()

	scores_explain = pred_model.predict(x_explain, verbose=0, batch_size=batch_size)[:,:,0]
	scores_explain = np.reshape(scores_explain, [scores_explain.shape[0], maxlen])

	if train:
		scores = pred_model.predict(x_val, verbose=0, batch_size=batch_size)[:, :, 0]
		scores = np.reshape(scores, [scores.shape[0], maxlen])

		pred_explain = model.predict(x_explain, verbose=0, batch_size=1)
		pred_fname_target_attack = 'pred_explain' + '-L2X_target' + '_k_' + str(args.k_words) + '_' + args.target_model + '.npy'
		np.save(os.path.join(args.outdir, pred_fname_target_attack), pred_explain)

		return scores, x_val, T, scores_explain, x_explain
	else:
		pred_explain = model.predict(x_explain, verbose=0, batch_size=batch_size)
		pred_fname_synon = args.save_fname_explain_id + '_pred_synon' + '-L2X_target' + '_k_' + str(
			args.k_words) + '_' + args.target_model + '.npy'
		np.save(os.path.join(args.outdir, pred_fname_synon), pred_explain)
		return T, scores_explain, x_explain


def save_scores(x, name, scores, k, args, fname):
	create_dataset_from_score(x, name, scores, k, args)
	np.save(os.path.join(args.outdir, fname), scores)
	# print('data/scores_val-L2X_'+args.dataset_name+'_k_' + str(k) + '_'+args.target_model+'.npy')
	if args.save_viz == "yes":
		save_sent_viz_file(x[:(x.shape[0] // 4)], name, scores[:(x.shape[0] // 4)], k, args)


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--task', type = str, choices=['original', 'L2X'], default = 'original')
	parser.add_argument('--train', action='store_true')

	parser.add_argument("--dataset_name",
						type=str,
						required=True,
						choices=['imdb', 'fake', 'yelp', 'ag', "mr", "amazon", "amazon_movies_small", "amazon_movies_20K"],
						help="")
	parser.add_argument("--target_model",
						type=str,
						required=True,
						choices=['wordLSTM', 'bert', 'wordCNN'],
						help="Target models for text classification: fasttext, charcnn, word level lstm "
							 "For NLI: InferSent, ESIM, bert-base-uncased")
	parser.add_argument("--target_model_path",
						type=str,
						required=True,
						help="pre-trained target model path")
	parser.add_argument("--test_data_path",
						type=str,
						help="Which dataset to Explain.")
	parser.add_argument("--train_data_path",
						type=str,
						help="Which dataset to train L2X on.")
	parser.add_argument("--target_data_path",
						type=str,
						required=True,
						help="Desired data that needed to be attacked.")
	parser.add_argument("--word_embeddings_path",
						type=str,
						default='',
						help="path to the word embeddings for CNN and LSTM models")
	parser.add_argument("--bert_tokenize",
						type=str,
						default="no",
						choices=['yes', 'true', 'no', 'false'],
						help="Do BERT tokenization")
	parser.add_argument("--nclasses",
						type=int,
						default=2,
						help="How many classes for classification.")
	parser.add_argument("--source_data_size",
						type=int,
						default=-1,
						help="")
	parser.add_argument("--batch_size",
						type=int,
						default=40,
						help="")
	parser.add_argument("--source_max_seq_length",
	                    default=128,
	                    type=int,
	                    help="max sequence length for source model")
	parser.add_argument("--target_max_seq_length",
	                    default=128,
	                    type=int,
	                    help="max sequence length for target model")
	parser.add_argument("--clean_test_data",
						type=str,
						default="yes",
						required=True,
						choices=['yes', 'true', 'no', 'false'],
						help="")
	parser.add_argument("--train_pred_labels",
						type=str,
						help="saved predicted training labels")
	parser.add_argument("--val_pred_labels",
						type=str,
						help="saved predicted validation labels")
	parser.add_argument("--save_fname_explain_id",
						type=str,
						default="",
						help="id to save explained scores")
	parser.add_argument("--k_words",
	                    default=20,
	                    type=int,
						required=True,
	                    help="How many words to learn to select")
	parser.add_argument("--seed",
	                    default=10086,
	                    type=int,
	                    help="")
	parser.add_argument("--outdir",
						type=str,
						default="data",
						help="output directory for scores")
	parser.add_argument("--save_viz",
						type=str,
						default="yes",
						choices=['yes', 'true', 'no', 'false'],
						help="saved predicted validation labels")

	parser.set_defaults(train=False)
	args = parser.parse_args()
	dict_a = vars(args)

	tf.set_random_seed(args.seed)
	np.random.seed(args.seed)
	print("Random seed = " + str(args.seed))

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
	config.log_device_placement = False  # to log device placement (on which device the operation ran)
	sess = tf.Session(config=config)
	set_session(sess)

	maxlen = args.source_max_seq_length
	assert maxlen >= args.target_max_seq_length
	print("Max lengths (l2x= "+str(maxlen)+", target= "+str(args.target_max_seq_length) + ")")
	batch_size = args.batch_size
	k = args.k_words

	if args.task == 'original':
		generate_original_preds(args.train)

	elif args.task == 'L2X':
		if args.train:
			scores_val, x, T, scores_explain, x_explain = L2X(args.train)
			print('Creating dataset with selected sentences...')
			# save_scores(x, "val", args.dataset_name, scores_val, k, args)
			dataset_name = "target"
			name = "explain"
			save_scores(x_explain, name, scores_explain, k, args,
						'scores_' + name + '-L2X_' + dataset_name + '_k_' + str(k) + '_' + args.target_model + '.npy')
		else:
			name = args.save_fname_explain_id+ "x_synon_explain_"
			T, scores_explain, x_explain = L2X(args.train)
			scores_fname_synon =  args.save_fname_explain_id + '_scores_synon' + '-L2X_target' + '_k_' + str(
				args.k_words) + '_' + args.target_model + '.npy'
			save_scores(x_explain, name, scores_explain, k, args, scores_fname_synon)

# # restore np.load for future normal usage
	# np.load = np_load_old




