try:
    from .l2x_tokenization import BertTokenizer
except:
    from l2x_tokenization import BertTokenizer
import argparse
try:
	import cPickle as pickle
except:
	import pickle

class Dataset_BERT:
    """
    Dataset class for Natural Language Inference datasets.

    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 pretrained_dir,
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
        # self.max_seq_length = max_seq_length
        self.batch_size = batch_size

def convert_examples_to_features(examples, max_seq_length, tokenizer, all_labels, label_list=["0", "1"], bert_tokenize=False):
    """Loads a data file into a list of `InputBatch`s."""

    new_all_labels = None
    if all_labels is not None:
        label_map = {label: i for i, label in enumerate(label_list)}
        new_all_labels = [label_map[label] for label in all_labels]

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(' '.join(example)) if bert_tokenize else example  # tokenizer.tokenize(' '.join(example))

        # tokens_b = None
        # if example.text_b:
        #     tokens_b = tokenizer.tokenize(example.text_b)
        #     # Modifies `tokens_a` and `tokens_b` in place so that the total
        #     # length is less than the specified length.
        #     # Account for [CLS], [SEP], [SEP] with "- 3"
        #     _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        # else:
        length_to_check = max_seq_length - 2 if bert_tokenize else max_seq_length
        if len(tokens_a) > length_to_check :  # max_seq_length - 2
            tokens_a = tokens_a[:(length_to_check)]  # (max_seq_length - 2)

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a
        if bert_tokenize is True:
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]   #TODO should not be needed for L2X selector
        segment_ids = [0] * len(tokens)

        # if tokens_b:
        #     tokens += tokens_b + ["[SEP]"]
        #     segment_ids += [1] * (len(tokens_b) + 1)

        """Converts a sequence of tokens into ids using the vocab."""

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
                      " sequence through BERT will result in indexing errors".format(len(ids), tokenizer.max_len))
            return ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens) if bert_tokenize else _convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        # input_mask += padding
        # segment_ids += padding

        assert len(input_ids) == max_seq_length
        # assert len(input_mask) == max_seq_length
        # assert len(segment_ids) == max_seq_length

        # label_id = label_map[example.label]
        if ex_index < 0:
            print("*** Example ***")
            # print("guid: %s" % (example.guid))
            print("tokens: %s" % " ".join([str(x) for x in tokens]))
            # print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            # print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            # print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            # print("label: %s (id = %d)" % (example.label, label_id))

        features.append(input_ids)
        #     InputFeatures(input_ids=input_ids,
        #                   input_mask=input_mask,
        #                   segment_ids=segment_ids,
        #                   label_id=label_id))
    # return features
    return features, new_all_labels

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

    # def convert_examples_to_features(self, examples, max_seq_length, tokenizer):
    #     """Loads a data file into a list of `InputBatch`s."""
    #
    #     features = []
    #     for (ex_index, text_a) in enumerate(examples):
    #         tokens_a = tokenizer.tokenize(' '.join(text_a))
    #
    #         # Account for [CLS] and [SEP] with "- 2"
    #         if len(tokens_a) > max_seq_length - 2:
    #             tokens_a = tokens_a[:(max_seq_length - 2)]
    #
    #         tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    #         segment_ids = [0] * len(tokens)
    #
    #         input_ids = tokenizer.convert_tokens_to_ids(tokens)
    #
    #         # The mask has 1 for real tokens and 0 for padding tokens. Only real
    #         # tokens are attended to.
    #         input_mask = [1] * len(input_ids)
    #
    #         # Zero-pad up to the sequence length.
    #         padding = [0] * (max_seq_length - len(input_ids))
    #         input_ids += padding
    #         input_mask += padding
    #         segment_ids += padding
    #
    #         assert len(input_ids) == max_seq_length
    #         assert len(input_mask) == max_seq_length
    #         assert len(segment_ids) == max_seq_length
    #
    #         features.append(
    #             InputFeatures(input_ids=input_ids,
    #                           input_mask=input_mask,
    #                           segment_ids=segment_ids))
    #     return features

    # def transform_text(self, data, batch_size=32):
    #     # transform data into seq of embeddings
    #     eval_features = self.convert_examples_to_features(data,
    #                                                       self.max_seq_length, self.tokenizer)
    #
    #     all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    #     all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    #     all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    #     eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    #
    #     # Run prediction for full data
    #     eval_sampler = SequentialSampler(eval_data)
    #     eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)
    #
    #     return eval_dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Required parameters
    # parser.add_argument("--dataset_path",
    #                     type=str,
    #                     required=True,
    #                     help="Which dataset to attack.")
    # parser.add_argument("--nclasses",
    #                     type=int,
    #                     default=2,
    #                     help="How many classes for classification.")
    parser.add_argument("--dataset_name",
                        type=str,
                        required=True,
                        choices=['imdb', 'fake', 'yelp'],
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
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Batch size to get prediction")
    parser.add_argument("--data_size",
                        default=-1,
                        type=int,
                        help="Data size to create adversaries")
    # parser.add_argument("--max_seq_length",
    #                     default=128,
    #                     type=int,
    #                     help="max sequence length for BERT target model")

    args = parser.parse_args()

    # dataset = Dataset_BERT(args.target_model_path, max_seq_length=max_seq_length, batch_size=args.batch_size)
    dataset = Dataset_BERT(args.target_model_path, batch_size=args.batch_size)

    # write python dict to a file
    output = open(args.dataset_name+'_vocab_'+args.target_model+'.pkl', 'wb')
    pickle.dump(dataset.tokenizer.vocab, output, protocol=0)
    output.close()

    # read python dict back from the file
    pkl_file = open(args.dataset_name+'_vocab_'+args.target_model+'.pkl', 'rb')
    my_vocab = pickle.load(pkl_file)
    pkl_file.close()

