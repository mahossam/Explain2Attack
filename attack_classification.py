import argparse
import glob, os
from builtins import str

import numpy as np
from future.backports.datetime import time

import dataloader
from train_classifier import Model
import criteria
import random

import tensorflow as tf
import tensorflow_hub as hub

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset

from BERT.tokenization import BertTokenizer
from BERT.modeling import BertForSequenceClassification, BertConfig

from bs4.dammit import EntitySubstitution
from tqdm import tqdm
import time, sys
from utils import str_to_bool
import subprocess
from datetime import datetime

class USE(object):
    def __init__(self, cache_path):
        super(USE, self).__init__()
        os.environ['TFHUB_CACHE_DIR'] = cache_path
        module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
        self.embed = hub.Module(module_url)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.build_graph()
        self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    def build_graph(self):
        self.sts_input1 = tf.placeholder(tf.string, shape=(None))
        self.sts_input2 = tf.placeholder(tf.string, shape=(None))

        sts_encode1 = tf.nn.l2_normalize(self.embed(self.sts_input1), axis=1)
        sts_encode2 = tf.nn.l2_normalize(self.embed(self.sts_input2), axis=1)
        self.cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
        clip_cosine_similarities = tf.clip_by_value(self.cosine_similarities, -1.0, 1.0)
        self.sim_scores = 1.0 - tf.acos(clip_cosine_similarities)

    def semantic_sim(self, sents1, sents2):
        scores = self.sess.run(
            [self.sim_scores],
            feed_dict={
                self.sts_input1: sents1,
                self.sts_input2: sents2,
            })
        return scores

def pick_most_similar_words_batch(src_words, sim_mat, idx2word, ret_count=10, threshold=0., reverse_output=False):
    """
    embeddings is a matrix with (d, vocab_size)
    """
    sim_order = np.argsort(-sim_mat[src_words, :])[:, 1:1 + ret_count]
    if reverse_output:
         sim_order = np.array([sim_order[i, ::-1] for i in range(len(sim_order))])
    sim_words, sim_values = [], []
    for idx, src_word in enumerate(src_words):
        sim_value = sim_mat[src_word][sim_order[idx]]
        mask = sim_value >= threshold
        sim_word, sim_value = sim_order[idx][mask], sim_value[mask]
        sim_word = [idx2word[id] for id in sim_word]
        sim_words.append(sim_word)
        sim_values.append(sim_value)
    return sim_words, sim_values


class NLI_infer_BERT(nn.Module):
    def __init__(self,
                 pretrained_dir,
                 nclasses,
                 max_seq_length=128,
                 batch_size=32):
        super(NLI_infer_BERT, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(pretrained_dir, num_labels=nclasses).cuda()

        # construct dataset loader
        # max_seq_length = max_seq_length + 2  # Account for additional [CLS] and [SEP] tokens of BERT (Not needed anymore after changing L2X data loading to exactly same as here)
        self.dataset = NLIDataset_BERT(pretrained_dir, max_seq_length=max_seq_length, batch_size=batch_size)

    def text_pred(self, text_data, batch_size=32, verbose=False, max_len=-1):
        # Switch the model to eval mode.
        self.model.eval()

        # transform text data into indices and create batches
        dataloader = self.dataset.transform_text(text_data, batch_size=batch_size)

        probs_all = []
        #         for input_ids, input_mask, segment_ids in tqdm(dataloader, desc="Evaluating"):
        # progb = tqdm(total=len(text_data))
        for input_ids, input_mask, segment_ids in dataloader:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)
            # if verbose:
            #     progb.update(1)
        # progb.close()
        return torch.cat(probs_all, dim=0)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class NLIDataset_BERT(Dataset):
    """
    Dataset class for Natural Language Inference datasets.

    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 pretrained_dir,
                 max_seq_length=128,
                 batch_size=32):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_premise_length: An integer indicating the maximum length
                accepted for the sequences in the premises. If set to None,
                the length of the longest premise in 'data' is used.
                Defaults to None.
            max_hypothesis_length: An integer indicating the maximum length
                accepted for the sequences in the hypotheses. If set to None,
                the length of the longest hypothesis in 'data' is used.
                Defaults to None.
        """
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_dir, do_lower_case=True)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.full_tokenization = True

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""

        features = []
        all_tokens = list()
        for (ex_index, text_a) in enumerate(examples):
            tokens_a = tokenizer.tokenize(' '.join(text_a)) if self.full_tokenization else text_a

            length_to_check = max_seq_length - 2 if self.full_tokenization else max_seq_length
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > length_to_check:
                tokens_a = tokens_a[:(length_to_check)]

            tokens = tokens_a
            if self.full_tokenization is True:
                tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            all_tokens.append(tokens)
            segment_ids = [0] * len(tokens)

            def _convert_tokens_to_ids(sentence):
                """Converts a sequence of tokens into ids using the vocab."""
                ids = []
                for token in sentence:
                    if token in tokenizer.vocab:
                        ids.append(tokenizer.vocab[token])
                    else:
                        ids.append(tokenizer.vocab[tokenizer.wordpiece_tokenizer.unk_token])
                if len(ids) > tokenizer.max_len:
                    print("warning: "
                        "Token indices sequence length is longer than the specified maximum "
                        " sequence length for this BERT model ({} > {}). Running this"
                        " sequence through BERT will result in indexing errors".format(len(ids), tokenizer.max_len) )
                return ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens) if self.full_tokenization else _convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
        return features, all_tokens

    def transform_text(self, data, batch_size=32):
        # transform data into seq of embeddings
        eval_features, _ = self.convert_examples_to_features(data, self.max_seq_length, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

        return eval_dataloader


def attack(text_ls, true_label, predictor, stop_words_set, word2idx, idx2word, cos_sim, sim_predictor=None,
           import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15, synonym_num=50,
           batch_size=32, l2x_score=None, args = None, bert_vocab = None, logfile=None, attack_dataset_name="", l2x_pred=None):
    # first check the prediction of the original text
    pred_max_length = args.attack_max_seq_length
    if args.target_model == "bert":
        pred_max_length = -1
    orig_probs = predictor([text_ls], max_len=pred_max_length).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()
    if args.word_rank_method == 'jin':
        # orig_prob = orig_probs.max()
        pass
    elif args.word_rank_method == 'l2x' or args.word_rank_method == 'hybrid':
        l2x_prob = np.max(l2x_pred)
        l2x_label = np.argmax(l2x_pred)

    if true_label != orig_label:
        return '', 0, orig_label, orig_label, 0, list(), -1.0
    else:
        len_text = len(text_ls)
        if len_text < sim_score_window:
            sim_score_threshold = 0.1  # shut down the similarity thresholding function
        half_sim_score_window = (sim_score_window - 1) // 2
        num_queries = 1

        # get the pos and verb tense info
        pos_ls = criteria.get_pos(text_ls)

        def jin_word_ranking():
            # get importance score
            leave_1_texts = [text_ls[:ii] + ['[UNK]' if args.target_model == 'bert' else '<oov>'] + text_ls[min(ii + 1, len_text):] for ii in range(len_text)]
            leave_1_probs = predictor(leave_1_texts, batch_size=batch_size, max_len=pred_max_length)
            leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)

            # get words to perturb ranked by importance scorefor word in words_perturb
            if str_to_bool(args.jin_argmax_based_i_score) is False:
                jin_import_scores = (
                            orig_prob - leave_1_probs[:, orig_label] + (leave_1_probs_argmax != orig_label).float() * (
                            leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0, leave_1_probs_argmax))).data.cpu().numpy()
            else:
                jin_import_scores = (
                            orig_prob - leave_1_probs[:, orig_label] + (leave_1_probs_argmax != orig_label).float() * (
                            leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0, leave_1_probs_argmax))).data.cpu().numpy()

            jin_words_perturb = []
            jin_sent_perturb_word_idx = []
            for idx, score in sorted(enumerate(jin_import_scores), key=lambda x: x[1], reverse=True):
                try:
                    if score > import_score_threshold and text_ls[idx] not in stop_words_set:
                        jin_words_perturb.append((idx, text_ls[idx]))
                        # if len(jin_sent_perturb_word_idx) < args.l2x_k_words:
                        jin_sent_perturb_word_idx.append(idx)
                except:
                    print(idx, len(text_ls), jin_import_scores.shape, text_ls, len(leave_1_texts))

            return jin_words_perturb, jin_sent_perturb_word_idx, len(leave_1_texts)

        def l2x_word_ranking():
            l2x_words_perturb = []
            l2x_sent_perturb_word_idx = []
            for idx, score in sorted(enumerate(l2x_score), key=lambda x: x[1], reverse=True):
                if str_to_bool(args.l2x_remove_stop_words) is True:
                    if text_ls[idx] not in stop_words_set:
                        l2x_words_perturb.append((idx, text_ls[idx]))
                        l2x_sent_perturb_word_idx.append(idx)
                else:
                    l2x_words_perturb.append((idx, text_ls[idx]))
                    l2x_sent_perturb_word_idx.append(idx)
                # if len(l2x_words_perturb) >= args.l2x_k_words:
                    # break

            return l2x_words_perturb, l2x_sent_perturb_word_idx

        if args.word_rank_method == 'jin':
            words_perturb, sent_perturb_word_idx, n_leave_1_texts = jin_word_ranking()
            num_queries += n_leave_1_texts
        elif args.word_rank_method == 'l2x':
            words_perturb, sent_perturb_word_idx = l2x_word_ranking()

        if args.word_rank_method == 'hybrid':
            if l2x_label == orig_label:
                words_perturb, sent_perturb_word_idx = l2x_word_ranking()
            else:
                words_perturb, sent_perturb_word_idx, n_leave_1_texts = jin_word_ranking()
                num_queries += n_leave_1_texts

        if str_to_bool(args.rank_only) is False:
            # find synonyms
            # words_perturb = words_perturb[:args.l2x_k_words] # Restrict to top K words to compare with L2X
            words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
            synonym_words, _ = pick_most_similar_words_batch(words_perturb_idx, cos_sim, idx2word, synonym_num, 0.5, reverse_output=str_to_bool(args.reverse_syn_order))
            synonyms_all = []
            for idx, word in words_perturb:
                if word in word2idx:
                    synonyms = synonym_words.pop(0)
                    if synonyms:
                        synonyms_all.append((idx, synonyms))

            # start replacing and attacking
            text_prime = text_ls[:]
            text_cache = text_prime[:]
            num_changed = 0

            syn_in_bert_vocab_ratio = 0.0
            unique_words_in_models_vocab = list()

            all_syns_flat = [syn for _, syns in synonyms_all for syn in syns]
            # logfile.write(f"syn_no={len(all_syns_flat)}")
            # print(f"syn_no={len(all_syns_flat)}")
            if args.target_model == "bert":
                unique_syns = list(set(all_syns_flat))
                unique_words_in_models_vocab = [syn for syn in unique_syns if syn in bert_vocab]
                syn_in_bert_vocab_ratio = len(unique_words_in_models_vocab) / len(unique_syns) if len(unique_syns) > 0 else 0.0

            for idx, synonyms in synonyms_all:
                new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]
                new_probs = predictor(new_texts, batch_size=batch_size, max_len=pred_max_length)
                current_synon_length = len(synonyms)

                # compute semantic similarity
                if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                    text_range_min = idx - half_sim_score_window
                    text_range_max = idx + half_sim_score_window + 1
                elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                    text_range_min = 0
                    text_range_max = sim_score_window
                elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
                    text_range_min = len_text - sim_score_window
                    text_range_max = len_text
                else:
                    text_range_min = 0
                    text_range_max = len_text
                semantic_sims = \
                    sim_predictor.semantic_sim([' '.join(text_cache[text_range_min:text_range_max])] * len(new_texts),
                                               list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts)))[0]
                num_queries += len(new_texts)
                if len(new_probs.shape) < 2:
                    new_probs = new_probs.unsqueeze(0)
                new_probs_mask = (orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
                # prevent bad synonyms
                new_probs_mask *= (semantic_sims >= sim_score_threshold)
                # prevent incompatible pos
                synonyms_pos_ls = [criteria.get_pos(new_text[max(idx - 4, 0):idx + 5])[min(4, idx)]
                                   if len(new_text) > 10 else criteria.get_pos(new_text)[idx] for new_text in new_texts]
                pos_mask = np.array(criteria.pos_filter(pos_ls[idx], synonyms_pos_ls))
                new_probs_mask *= pos_mask

                if np.sum(new_probs_mask) > 0:
                    text_prime[idx] = synonyms[(new_probs_mask * semantic_sims).argmax()]
                    num_changed += 1
                    break
                else:
                    new_label_probs = new_probs[:, orig_label] + torch.from_numpy(
                        (semantic_sims < sim_score_threshold) + (1 - pos_mask).astype(float)).float().cuda()
                    if str_to_bool(args.reverse_syn_order):
                        new_label_prob_min, new_label_prob_argmin = new_label_probs[0], 0
                    else:
                        new_label_prob_min, new_label_prob_argmin = torch.min(new_label_probs, dim=-1)
                    if new_label_prob_min < orig_prob:
                        text_prime[idx] = synonyms[new_label_prob_argmin]
                        num_changed += 1
                text_cache = text_prime[:]

            return ' '.join(text_prime), num_changed, orig_label, torch.argmax(
                predictor([text_prime], max_len=pred_max_length)), num_queries, sent_perturb_word_idx, syn_in_bert_vocab_ratio
        else:
            return " ", 0, orig_label, orig_label, num_queries, sent_perturb_word_idx, 0.0


def random_attack(text_ls, true_label, predictor, perturb_ratio, stop_words_set, word2idx, idx2word, cos_sim,
                  sim_predictor=None, import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15,
                  synonym_num=50, batch_size=32, args=None):
    # first check the prediction of the original text
    pred_max_length = args.attack_max_seq_length
    if args.target_model == "bert":
        pred_max_length = -1
    orig_probs = predictor([text_ls], max_len=pred_max_length).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()
    if true_label != orig_label:
        return '', 0, orig_label, orig_label, 0
    else:
        len_text = len(text_ls)
        if len_text < sim_score_window:
            sim_score_threshold = 0.1  # shut down the similarity thresholding function
        half_sim_score_window = (sim_score_window - 1) // 2
        num_queries = 1

        # get the pos and verb tense info
        pos_ls = criteria.get_pos(text_ls)

        # randomly get perturbed words
        perturb_idxes = random.sample(range(len_text), int(len_text * perturb_ratio))
        words_perturb = [(idx, text_ls[idx]) for idx in perturb_idxes]

        # find synonyms
        words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
        synonym_words, _ = pick_most_similar_words_batch(words_perturb_idx, cos_sim, idx2word, synonym_num, 0.5)
        synonyms_all = []
        for idx, word in words_perturb:
            if word in word2idx:
                synonyms = synonym_words.pop(0)
                if synonyms:
                    synonyms_all.append((idx, synonyms))

        # start replacing and attacking
        text_prime = text_ls[:]
        text_cache = text_prime[:]
        num_changed = 0
        for idx, synonyms in synonyms_all:
            new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]
            new_probs = predictor(new_texts, batch_size=batch_size, max_len=pred_max_length)

            # compute semantic similarity
            if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = idx - half_sim_score_window
                text_range_max = idx + half_sim_score_window + 1
            elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = 0
                text_range_max = sim_score_window
            elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
                text_range_min = len_text - sim_score_window
                text_range_max = len_text
            else:
                text_range_min = 0
                text_range_max = len_text
            semantic_sims = \
            sim_predictor.semantic_sim([' '.join(text_cache[text_range_min:text_range_max])] * len(new_texts),
                                       list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts)))[0]

            num_queries += len(new_texts)
            if len(new_probs.shape) < 2:
                new_probs = new_probs.unsqueeze(0)
            new_probs_mask = (orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
            # prevent bad synonyms
            new_probs_mask *= (semantic_sims >= sim_score_threshold)
            # prevent incompatible pos
            synonyms_pos_ls = [criteria.get_pos(new_text[max(idx - 4, 0):idx + 5])[min(4, idx)]
                               if len(new_text) > 10 else criteria.get_pos(new_text)[idx] for new_text in new_texts]
            pos_mask = np.array(criteria.pos_filter(pos_ls[idx], synonyms_pos_ls))
            new_probs_mask *= pos_mask

            if np.sum(new_probs_mask) > 0:
                text_prime[idx] = synonyms[(new_probs_mask * semantic_sims).argmax()]
                num_changed += 1
                break
            else:
                new_label_probs = new_probs[:, orig_label] + torch.from_numpy(
                        (semantic_sims < sim_score_threshold) + (1 - pos_mask).astype(float)).float().cuda()
                new_label_prob_min, new_label_prob_argmin = torch.min(new_label_probs, dim=-1)
                if new_label_prob_min < orig_prob:
                    text_prime[idx] = synonyms[new_label_prob_argmin]
                    num_changed += 1
            text_cache = text_prime[:]
        return ' '.join(text_prime), num_changed, orig_label, torch.argmax(predictor([text_prime], max_len=pred_max_length)), num_queries


def remove_padding_if_any(texts):
    new_texts = list()
    for sent in texts:
        new_sent = list()
        for w in sent:
            if w != "<PAD>" and w != "[PAD]" and w != "<pad>":
                new_sent.append(w)
        new_texts.append(new_sent)
    return new_texts


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--attack_data_path",
                        type=str,
                        required=True,
                        help="Which dataset to attack.")
    parser.add_argument("--l2x_test_data_path",
                        type=str,
                        help="Which dataset to validate L2X on.")
    parser.add_argument("--l2x_train_data_path",
                        type=str,
                        help="Which dataset to train L2X on.")
    parser.add_argument("--nclasses",
                        type=int,
                        default=2,
                        help="How many classes for classification.")
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
    parser.add_argument("--word_embeddings_path",
                        type=str,
                        default='',
                        help="path to the word embeddings for the target model")
    parser.add_argument("--counter_fitting_embeddings_path",
                        type=str,
                        required=True,
                        help="path to the counter-fitting embeddings we used to find synonyms")
    parser.add_argument("--counter_fitting_cos_sim_path",
                        type=str,
                        default='',
                        help="pre-compute the cosine similarity scores based on the counter-fitting embeddings")
    parser.add_argument("--USE_cache_path",
                        type=str,
                        required=True,
                        help="Path to the USE encoder cache.")
    parser.add_argument("--output_dir",
                        type=str,
                        default='adv_results',
                        help="The output directory where the attack results will be written.")
    parser.add_argument("--outdir_postfix",
                        type=str,
                        default='', required=True,
                        help=".")
    parser.add_argument("--word_rank_method",
                        type=str,
                        required=True,
                        choices=['jin', 'l2x', 'hybrid'],
                        help="The word importance ranking method to be used")
    parser.add_argument("--jin_argmax_based_i_score",
                        type=str,
                        default="no",
                        choices=['yes', 'true', 'no', 'false'],
                        help="")
    parser.add_argument("--l2x_k_words",
                        type=int,
                        default=20,
                        help="K words to select for L2X module.")
    parser.add_argument("--l2x_bert_tokenize",
                        type=str,
                        default="no",
                        choices=['yes', 'true', 'no', 'false'],
                        help="Do BERT tokenization")
    parser.add_argument("--l2x_remove_stop_words",
                        type=str,
                        default="yes",
                        required=True,
                        choices=['yes', 'no', 'true', 'false'],
                        help="Keep or remove english stopwords for L2X")
    parser.add_argument("--rank_only",
                        type=str,
                        default="no",
                        choices=['yes', 'true', 'no', 'false'],
                        help="Don't attack, only rank words")
    parser.add_argument("--clean_test_data",
                        type=str,
                        default="yes",
                        choices=['yes', 'true', 'no', 'false'],
                        help="")
    parser.add_argument("--data_format_label_fst",
                        type=str,
                        default="yes",
                        choices=['yes', 'true', 'no', 'false'],
                        help="")
    # parser.add_argument("--remove_all_probs",
    #                     type=str,
    #                     required=True,
    #                     choices=['yes', 'true', 'no', 'false'],
    #                     help="Don't use any target model probs, only labels")

    ## Model hyperparameters
    parser.add_argument("--sim_score_window",
                        default=15,
                        type=int,
                        help="Text length or token number to compute the semantic similarity score")
    parser.add_argument("--import_score_threshold",
                        default=-1.,
                        type=float,
                        help="Required mininum importance score.")
    parser.add_argument("--sim_score_threshold",
                        default=0.7,
                        type=float,
                        help="Required minimum semantic similarity score.")
    parser.add_argument("--synonym_num",
                        default=50,
                        type=int,
                        help="Number of synonyms to extract")
    parser.add_argument("--reverse_syn_order",
                        type=str,
                        default="no",
                        choices=['yes', 'true', 'no', 'false'],
                        help="")
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Batch size to get prediction")
    parser.add_argument("--l2x_batch_size",
                        default=40,
                        type=int,
                        help="")
    parser.add_argument("--attack_data_size",
                        default=1000,
                        type=int,
                        help="Data size to create adversaries")
    parser.add_argument("--l2x_test_data_size",
                        default=1000000,
                        type=int,
                        help="")
    parser.add_argument("--l2x_train_data_size",
                        default=10000000,
                        type=int,
                        help="")
    parser.add_argument("--perturb_ratio",
                        default=0.,
                        type=float,
                        help="Whether use random perturbation for ablation study")
    parser.add_argument("--l2x_max_seq_length",
                        default=128,
                        type=int,
                        help="max sequence length for BERT target model")
    parser.add_argument("--l2x_rseed",
                        default=10086,
                        type=int,
                        help="")
    parser.add_argument("--attack_max_seq_length",
                        default=128,
                        type=int,
                        required=True,
                        help="max sequence length for target model")
    parser.add_argument("--r_pos",
                        default=-1,
                        type=int,
                        help="position to start resuming the script after interruption")
    parser.add_argument("--r_path",
                        type=str,
                        default='',
                        help="path to start resuming the script after interruption")


    args = parser.parse_args()
    if args.r_pos >= 0 and args.r_path == '':
        print("Must provide resume path")
        return

    attack_data_name = os.path.split(os.path.dirname(args.attack_data_path))[-1]
    l2x_data_name = ""
    if args.word_rank_method == 'l2x' or args.word_rank_method == 'hybrid':
        l2x_data_name = os.path.split(os.path.dirname(args.l2x_train_data_path))[-1]
        # assert args.l2x_test_data_size >= args.attack_data_size

    if args.r_path == '':
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S%f")
        run_id = timestamp + "_" + args.outdir_postfix + "_" + f'_{args.word_rank_method}'
        if args.word_rank_method == "l2x" or args.word_rank_method == "hybrid":
            run_id += f'_k_{args.l2x_k_words}'
        if args.word_rank_method == "jin" or args.word_rank_method == "hybrid":
            run_id += f'_max_len_{args.attack_max_seq_length}'
        args.output_dir = os.path.join(args.output_dir, args.target_model, attack_data_name, run_id)
        if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
            print("Output directory ({}) already exists and is not empty.".format(args.output_dir))
        else:
            os.makedirs(args.output_dir, exist_ok=True)
    else:
        args.output_dir = args.r_path

    log_file = open(os.path.join(args.output_dir, f'results_log_{args.target_model}_{args.word_rank_method}'), 'a')
    log_file.write("Run with command:\n" + " ".join([arg for arg in sys.argv[1:]])+"\n")
    print("output dir.: "+args.output_dir)
    log_file.write("output dir.: "+args.output_dir+"\n")
    log_file.flush()

    if args.r_pos <= -1 :
        if os.path.exists(os.path.join(args.output_dir, f'attack_idx.npy')):
            resume_position = int(np.load(os.path.join(args.output_dir, f'attack_idx.npy')))
            print(f"resuming from attack_idx.npy pos = {resume_position}")
            log_file.write(f"resuming from attack_idx.npy pos = {resume_position}\n")
            log_file.flush()
        else:
            resume_position = 0
    else:
        resume_position = args.r_pos


    if (args.word_rank_method == 'l2x' or args.word_rank_method == 'hybrid') and args.r_path == '':
        _, labels_train = dataloader.read_corpus(args.l2x_train_data_path, clean=False, lower=True, max_length=args.attack_max_seq_length, labels_only=True)
        _, _labels_test = dataloader.read_corpus(args.l2x_test_data_path, clean=str_to_bool(args.clean_test_data), lower=True, max_length=args.attack_max_seq_length, labels_only=True)
        labels_train, _labels_test = labels_train[:args.l2x_train_data_size], _labels_test[:args.l2x_test_data_size]
        if 0 not in _labels_test:
            _labels_test = [l - 1 for l in _labels_test]
        print("Test labels = " + str(list(set(_labels_test))))
        log_file.write("Test labels = " + str(list(set(_labels_test)))+"\n")
        if 0 not in labels_train:
            labels_train = [l - 1 for l in labels_train]
        print("Train labels = " + str(list(set(labels_train))))
        log_file.write("Train labels = " + str(list(set(labels_train)))+"\n")

        ntrain, ntest = len(labels_train), len(_labels_test)
        labels_train = labels_train[:(ntrain - ntrain % args.l2x_batch_size)]
        labels_test = _labels_test[:(ntest - ntest % args.l2x_batch_size)]

        print("L2X data train and test import finished!")

    # get data to attack
    load_max_len = -1
    if args.target_model != "bert":
        load_max_len = args.attack_max_seq_length  # BERT already trims input sentences to its max length
    fix_labels = False
    if (attack_data_name == "yelp" or attack_data_name == "fake" or attack_data_name == "ag") and not "author" in args.attack_data_path:
        fix_labels = True
    texts, labels = dataloader.read_corpus(args.attack_data_path, clean=str_to_bool(args.clean_test_data), lower=True, fix_labels=fix_labels, max_length=load_max_len)  # TODO lower should = False if input contains spetial chars like [UNK] or [PAD]
    # if 0 not in labels:
    #     labels = [l - 1 for l in labels]
    print("Test labels = ", list(set(labels)))
    log_file.write("Test labels = " + str(list(set(labels)))+"\n")
    data = list(zip(texts, labels))
    data = data[:args.attack_data_size]  # choose how many samples for adversary
    print("Attack data import finished!")

    log_file.flush()

    bert_vocab = None
    # construct the modellstm
    print("Building Model...")
    if args.target_model == 'wordLSTM':
        model = Model(args.word_embeddings_path, nclasses=args.nclasses, hidden_size=100).cuda()  # TODO change input format to <oov> <pad> for case of LSTM or CNN classif, [UNK] and [PAD] in case of bert, or _OOV_, _PAD_ for case of NLI
        checkpoint = torch.load(args.target_model_path, map_location='cuda:0')
        model.load_state_dict(checkpoint)
    elif args.target_model == 'wordCNN':
        model = Model(args.word_embeddings_path, nclasses=args.nclasses, hidden_size=100, cnn=True).cuda()  # TODO change input format to <oov> <pad> for case of LSTM or CNN classif, [UNK] and [PAD] in case of bert, or _OOV_, _PAD_ for case of NLI
        checkpoint = torch.load(args.target_model_path, map_location='cuda:0')
        model.load_state_dict(checkpoint)
    elif args.target_model == 'bert':
        model = NLI_infer_BERT(args.target_model_path, nclasses=args.nclasses, max_seq_length=args.attack_max_seq_length)
        bert_vocab = model.dataset.tokenizer.vocab
    predictor = model.text_pred

    if (args.word_rank_method == 'l2x'  or args.word_rank_method == 'hybrid') and args.target_model == 'bert' and str_to_bool(args.l2x_bert_tokenize) is True and args.r_path == '':
        # model.dataset.max_seq_length = args.l2x_max_seq_length
        # _, texts_test = model.dataset.convert_examples_to_features(texts_test, args.l2x_max_seq_length, model.dataset.tokenizer)
        # _, texts_train = model.dataset.convert_examples_to_features(texts_train, args.l2x_max_seq_length, model.dataset.tokenizer)
        #
        # # texts are for attacking (they don't need to be adjusted to l2x train batch_size)
        # _, texts = model.dataset.convert_examples_to_features(texts, args.attack_max_seq_length, model.dataset.tokenizer)
        # # texts, labels = _texts_test, _labels_test
        # model.dataset.full_tokenization = False
        # data = list(zip(texts, labels))
        #
        # model.dataset.max_seq_length = args.attack_max_seq_length
        # print("Data test tokenized for BERT!")
        pass

    if (args.word_rank_method == 'l2x' or args.word_rank_method == 'hybrid') and args.r_path == '':
        # model.dataset.max_seq_length = args.l2x_max_seq_length

        save_fname_train = l2x_data_name + '_' + os.path.splitext(os.path.basename(args.l2x_train_data_path))[
            0] + '_' + args.target_model
        save_all_labels(args.output_dir, save_fname_train, labels_train, args)
        del labels_train  # free up some memory

        # model.dataset.max_seq_length = args.attack_max_seq_length

    print("Model built!")

    # prepare synonym extractor
    # build dictionary via the embedding file
    idx2word = {}
    word2idx = {}

    print("Building vocab...")
    with open(args.counter_fitting_embeddings_path, 'r') as ifile:
        for line in ifile:
            word = line.split()[0]
            if word not in idx2word:
                idx2word[len(idx2word)] = word
                word2idx[word] = len(idx2word) - 1

    print("Building cos sim matrix...")
    if args.counter_fitting_cos_sim_path:
        # load pre-computed cosine similarity matrix if provided
        print('Load pre-computed cosine similarity matrix from {}'.format(args.counter_fitting_cos_sim_path))
        cos_sim = np.load(args.counter_fitting_cos_sim_path)
    else:
        # calculate the cosine similarity matrix
        print('Start computing the cosine similarity matrix!')
        embeddings = []
        with open(args.counter_fitting_embeddings_path, 'r') as ifile:
            for line in ifile:
                embedding = [float(num) for num in line.strip().split()[1:]]
                embeddings.append(embedding)
        embeddings = np.array(embeddings)
        product = np.dot(embeddings, embeddings.T)
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        cos_sim = product / np.dot(norm, norm.T)
    print("Cos sim import finished!")

    # build the semantic similarity module
    use = USE(args.USE_cache_path)

    # start attacking
    # if args.r_path == '':
    orig_failures = 0.
    adv_failures = 0.
    changed_rates = []
    nums_queries = []
    nums_changed = list()
    orig_texts = []
    adv_texts = []
    true_labels = []
    new_labels = []
    all_importance_scores, all_syn_in_bert_vocab_ratios = list(), list()
    # else:
    #     orig_failures = float(np.load(os.path.join(args.output_dir, f'orig_failures.npy')))
    #     adv_failures = float(np.load(os.path.join(args.output_dir, f'adv_failures.npy')))
    #     changed_rates = np.load(os.path.join(args.output_dir, f'changed_rates.npy'))#[:resume_position+1]
    #     nums_queries = np.load(os.path.join(args.output_dir, f'nums_queries.npy'))#[:resume_position+1]
    #     true_labels = np.load(os.path.join(args.output_dir, f'true_labels.npy'))#[:resume_position+1]
    #     new_labels = np.load(os.path.join(args.output_dir, f'new_labels.npy'))#[:resume_position+1]
    #     all_syn_in_bert_vocab_ratios= np.load(os.path.join(args.output_dir, f'all_syn_in_bert_vocab_ratios.npy'))
    #     nums_changed = np.load(os.path.join(args.output_dir, f'{args.word_rank_method}_{args.target_model}_count_changed_words_per_sent.npy'))

    update_every = 250
    stop_words_set = criteria.get_stopwords()

    l2x_scores = None
    if args.word_rank_method == 'l2x' or args.word_rank_method == 'hybrid':
        if args.r_path == '':
            # model.dataset.max_seq_length = args.l2x_max_seq_length
            save_fname_test = l2x_data_name + '_' + os.path.splitext(os.path.basename(args.l2x_test_data_path))[0] + '_' + args.target_model
            # if str_to_bool(args.l2x_bert_tokenize) is True:
            #     model.dataset.full_tokenization = False
            save_all_labels(args.output_dir, save_fname_test, labels_test, args)
            # acc = np.mean(np.argmax(probs_test.cpu().numpy(), axis=1) == np.array(labels_test))
            # print(f'The val accuracy of the target model is {acc}')
            # log_file.writelines(f'The val accuracy of the target model is {acc}')
            arguments = ["./run_l2x.sh", "mahmoud", "--task=L2X", "--train",  f'--dataset_name={l2x_data_name}', f'--k_words={args.l2x_k_words}',
                             f'--nclasses={args.nclasses}', f'--target_model={args.target_model}', f'--target_model_path={args.target_model_path}',
                             f'--train_data_path={args.l2x_train_data_path}', f'--test_data_path={args.l2x_test_data_path}',
                                  f'--target_data_path={args.attack_data_path}',
                                  f'--source_max_seq_length={args.l2x_max_seq_length}', f'--target_max_seq_length={args.l2x_max_seq_length if args.target_model == "bert" else args.attack_max_seq_length}',
                                  f'--source_data_size={args.l2x_train_data_size}', f'--batch_size={args.l2x_batch_size}',
                                  f'--clean_test_data={args.clean_test_data}',
                                  '--train_pred_labels=' + os.path.join(os.path.abspath(args.output_dir), "labels_" + save_fname_train + '.npy'),
                                  '--val_pred_labels=' + os.path.join(os.path.abspath(args.output_dir), "labels_" + save_fname_test + '.npy'),
                                  f'--bert_tokenize={args.l2x_bert_tokenize}', f'--seed={args.l2x_rseed}',
                                  f'--outdir={os.path.abspath(args.output_dir)}', '--save_viz=no']
            arguments += [f'--word_embeddings_path={args.word_embeddings_path}'] if (args.target_model == "wordCNN" or args.target_model == "wordLSTM") else list()
            res = subprocess.run(arguments, stderr=sys.stdout, stdout=sys.stdout)  #, stderr=subprocess.PIPE, stdout=subprocess.PIPE)  # , stderr=sys.stdout, stdout=sys.stdout)
        scores_fname = "scores_explain-L2X_target" + '_k_' + str(args.l2x_k_words) + '_' + args.target_model + '.npy'
        pred_fname = "pred_explain-L2X_target" + '_k_' + str(args.l2x_k_words) + '_' + args.target_model + '.npy'
        l2x_scores = np.load(os.path.join(args.output_dir, scores_fname))
        l2x_scores = l2x_scores[:args.attack_data_size]

        l2x_preds = np.load(os.path.join(args.output_dir, pred_fname))
        l2x_preds = l2x_preds[:args.attack_data_size]
        # model.dataset.max_seq_length = args.attack_max_seq_length

    print('Start attacking!') if args.r_path == '' else print('Resume attacking (using saved L2X scores)!')
    pbar = tqdm(total=len(data))
    pbar.update(max(0, resume_position-1))
    for idx, (text, true_label) in enumerate(data):
        if idx >= resume_position:
            np.save(os.path.join(args.output_dir, f'attack_and_sublists_idx.npy'), np.array(
                [idx, len(all_importance_scores), len(all_syn_in_bert_vocab_ratios), len(nums_queries),
                 len(nums_changed), len(changed_rates), len(new_labels)]))
            if idx % update_every == 0 and idx != 0:
                pbar.update(update_every)
                log_file.write(str(pbar) + '\n')
                log_file.flush()
                log_current_results(log_file, idx, orig_failures, adv_failures, changed_rates, nums_changed, nums_queries, orig_texts, adv_texts, true_labels, new_labels, all_syn_in_bert_vocab_ratios, args)
                save_sent_viz_file(data[:len(all_importance_scores)], all_importance_scores, args.l2x_k_words, args)
            if args.perturb_ratio > 0.:
                new_text, num_changed, orig_label, \
                new_label, num_queries = random_attack(text, true_label, predictor, args.perturb_ratio, stop_words_set,
                                                        word2idx, idx2word, cos_sim, sim_predictor=use,
                                                        sim_score_threshold=args.sim_score_threshold,
                                                        import_score_threshold=args.import_score_threshold,
                                                        sim_score_window=args.sim_score_window,
                                                        synonym_num=args.synonym_num,
                                                        batch_size=args.batch_size, args=args)
            else:
                single_l2x_score = None
                if "scores_len_eq_text" in args.outdir_postfix:
                    single_l2x_score = l2x_scores[idx, :len(text)] if args.word_rank_method == 'l2x' or args.word_rank_method == 'hybrid' else None
                if "scores_len_max_attack_len" in args.outdir_postfix:
                    l2x_max_length = args.l2x_max_seq_length if args.target_model == "bert" else args.attack_max_seq_length
                    single_l2x_score = l2x_scores[idx, :min(len(text), l2x_max_length)] if args.word_rank_method == 'l2x' or args.word_rank_method == 'hybrid' else None
                if "scores_full" in args.outdir_postfix:
                    single_l2x_score = l2x_scores[idx] if args.word_rank_method == 'l2x' or args.word_rank_method == 'hybrid' else None
                single_l2x_pred = l2x_preds[idx] if args.word_rank_method == 'l2x' or args.word_rank_method == 'hybrid' else None
                new_text, num_changed, orig_label, \
                new_label, num_queries, sent_perturb_word_idx, syn_in_bert_vocab_ratio = attack(text, true_label, predictor, stop_words_set,
                                                word2idx, idx2word, cos_sim, sim_predictor=use,
                                                sim_score_threshold=args.sim_score_threshold,
                                                import_score_threshold=args.import_score_threshold,
                                                sim_score_window=args.sim_score_window,
                                                synonym_num=args.synonym_num,
                                                batch_size=args.batch_size,
                                                                       l2x_score=single_l2x_score, args=args, bert_vocab=bert_vocab, logfile=log_file, attack_dataset_name=attack_data_name, l2x_pred=single_l2x_pred)

                all_importance_scores.append(sent_perturb_word_idx)
                all_syn_in_bert_vocab_ratios.append(syn_in_bert_vocab_ratio)

            if true_label != orig_label:
                orig_failures += 1
            else:
                nums_queries.append(num_queries)
                nums_changed.append(num_changed)
            if true_label != new_label:
                adv_failures += 1

            changed_rate = 1.0 * num_changed / len(text)

            if true_label == orig_label and true_label != new_label:
                changed_rates.append(changed_rate)
                orig_texts.append(' '.join(text))
                adv_texts.append(new_text)
                true_labels.append(true_label)
                new_labels.append(new_label)

    pbar.update(pbar.total - pbar.n)
    log_file.write(str(pbar) + '\n')
    log_file.flush()
    pbar.close()
    save_sent_viz_file(data, all_importance_scores, args.l2x_k_words, args)

    true_data_size = len(data)
    _summary = f'Data size = {true_data_size}, (Cln==Orig) = {true_data_size - orig_failures}, (Cln=Orig and Cln!=NewAdv) = {len(adv_texts)}, ({len(adv_texts)/true_data_size*100.0}% of all data, {(1-(len(adv_texts)/true_data_size))*100.0} % attack-acc.)\n'
    print(_summary)
    log_file.write(_summary)
    log_current_results(log_file, true_data_size, orig_failures, adv_failures, changed_rates, nums_changed, nums_queries, orig_texts, adv_texts, true_labels, new_labels, all_syn_in_bert_vocab_ratios, args)
    log_file.close()


def log_current_results(log_file, current_data_size, orig_failures, adv_failures, changed_rates, nums_changed,
                        nums_queries, orig_texts, adv_texts, true_labels, new_labels, all_syn_in_bert_vocab_ratios, args, resume=False):
    message = 'For target model {}: original accuracy: {:.3f}%, adv accuracy: {:.3f}%, ' \
              'changed rate: avg: {:.3f}%, changed words avg:{:.3f}  max: {:d}  min: {:d} , avg num of queries: {:.1f}\n'.format(
        args.target_model,
        (1 - orig_failures / current_data_size) * 100,
        (1 - adv_failures / current_data_size) * 100,
        np.mean(changed_rates) * 100, np.mean(nums_changed),
        np.max(nums_changed),
        np.min(nums_changed),
        np.mean(nums_queries))
    message += f"\nall_syn_in_bert_vocab_ratios = {np.mean([ratio for ratio in all_syn_in_bert_vocab_ratios if ratio >= 0.0])}\n"

    # save generated data
    np.save(os.path.join(args.output_dir, f'{args.word_rank_method}_{args.target_model}_count_changed_words_per_sent.npy'),
            np.array(nums_changed))
    np.save(os.path.join(args.output_dir, f'orig_failures.npy'), orig_failures)
    np.save(os.path.join(args.output_dir, f'adv_failures.npy'), adv_failures)
    np.save(os.path.join(args.output_dir, f'changed_rates.npy'), np.array(changed_rates))
    np.save(os.path.join(args.output_dir, f'nums_queries.npy'), np.array(nums_queries))
    np.save(os.path.join(args.output_dir, f'true_labels.npy'), np.array(true_labels))
    np.save(os.path.join(args.output_dir, f'new_labels.npy'), np.array(new_labels))
    np.save(os.path.join(args.output_dir, f'all_syn_in_bert_vocab_ratios.npy'), np.array(all_syn_in_bert_vocab_ratios))

    print(message)
    log_file.write(f'orig_failures = {orig_failures}, adv_failures = {adv_failures}\n')
    log_file.write(message)
    log_file.flush()

    with open(os.path.join(args.output_dir, f'orig_texts.txt'), 'w') as ofile:
        for tx in orig_texts:
            ofile.write(tx+"\n")
        ofile.flush()
    with open(os.path.join(args.output_dir, f'adv_texts.txt'), 'w') as ofile:
        for tx in adv_texts:
            ofile.write(tx+"\n")
        ofile.flush()
    with open(os.path.join(args.output_dir, f'adversaries_{args.target_model}_{args.word_rank_method}.txt'), 'w') as ofile:
        for orig_text, adv_text, true_label, new_label in zip(orig_texts, adv_texts, true_labels, new_labels):
            ofile.write('orig sent ({}):\t{}\nadv sent ({}):\t{}\n\n'.format(true_label, orig_text, new_label, adv_text))
        ofile.flush()


def save_sent_viz_file(x_y_pairs, scores, k, args):
    escaper = EntitySubstitution()

    with open(os.path.join(args.output_dir, f'sent_viz_top_few_{args.target_model}_{args.word_rank_method}.html'), 'w') as txt_file:
        txt_file.write("<!DOCTYPE html>\n<html>\n<body>\n")

        for i, (x_single, label) in enumerate(x_y_pairs):

            sent_viz = list()
            for o_wi, word in enumerate(x_single):
                if o_wi in scores[i]:  # x_selected[s_i:][wp] != 0:
                    placeholder = u"<mark><strong>" + escaper.substitute_html(word) + u"</strong></mark>"
                else:
                    placeholder = escaper.substitute_html(word)
                sent_viz.append(placeholder)

            txt_file.write((u"<p>(" + str(label) + u") " + u" ".join(sent_viz) + u"</p><br>\n"))
        txt_file.write("</body>\n</html>\n")

    try:
        word_ranking = np.array(scores)
        np.save(os.path.join(args.output_dir, f'word_ranking_{args.target_model}_{args.word_rank_method}'), word_ranking)
    except:
        print("Coulnd't create numpy array of word ranking")

    # with  open(os.path.join(args.output_dir, f'sent_viz_{args.target_model}_{args.word_rank_method}.html'), 'w') as scores:


def save_all_predictionas_probs_and_labels(outdir, fname, predictor, texts, args):
    pred_max_length = args.attack_max_seq_length
    if args.target_model == "bert":
        pred_max_length = -1
    probs = predictor(texts, verbose=True, max_len=pred_max_length).squeeze()
    array_to_save_probs = probs.cpu().numpy()
    labels = torch.argmax(probs, dim=1)
    array_to_save_labels = np.eye(args.nclasses)[labels.cpu().numpy()]
    # np.save(os.path.join(outdir, 'pred_' + fname), array_to_save_probs)
    np.save(os.path.join(outdir, 'labels_' + fname), array_to_save_labels)
    return probs

def save_all_labels(outdir, fname, labels, args):
    array_to_save_labels = np.eye(args.nclasses)[labels]
    # np.save(os.path.join(outdir, 'pred_' + fname), array_to_save_probs)
    np.save(os.path.join(outdir, 'labels_' + fname), array_to_save_labels)



if __name__ == "__main__":
    main()
