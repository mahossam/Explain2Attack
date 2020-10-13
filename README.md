# Transfer2Attack


## Prerequisites:
* Pytorch >= 0.4
* Tensorflow >= 1.0 
* Numpy
* Python >= 3.6

## How to use

* Run the following code to install the **esim** package:

 ```
cd ESIM
python setup.py install
cd ..
```
* Please follow `download_instructions.txt` inside subfolders to download missing data files.
* (Optional) Run the following code to pre-compute the cosine similarity scores between word pairs based on the [counter-fitting word embeddings](https://github.com/nmrksic/counter-fitting).

```
python comp_cos_sim_mat.py [PATH_TO_COUNTER_FITTING_WORD_EMBEDDINGS]
```

* Run the following code to generate the adversaries for text classification:

```
python attack_classification.py
```

Example to reproduce IMDB results:
```
python attack_classification.py --l2x_train_data_size 25000 --l2x_train_data_path data/amazon_movies_20K/train_tok.csv --l2x_test_data_size 25000 --l2x_test_data_path data/amazon_movies_20K/test_tok.csv --l2x_max_seq_length 256 --l2x_k_words 20 --l2x_bert_tokenize no --attack_data_size 25000 --data_format_label_fst yes --attack_data_path data/imdb/test_tok.csv --word_rank_method l2x --target_model wordLSTM --target_model_path classifiers/lstm/imdb_trim_len_256_0 --word_embeddings_path data/embeddings/glove.6B/glove.6B.200d.txt --attack_max_seq_length 256 --batch_size 32 --counter_fitting_embeddings_path counter-fitting/word_vectors/counter-fitted-vectors.txt --counter_fitting_cos_sim_path ./cos_sim_counter_fitting.npy --USE_cache_path ./tf_cache --l2x_remove_stop_words yes --rank_only no --l2x_rseed 10086 --sim_score_threshold 0.7 --outdir_postfix scores_len_max_attack_len
```

This code is based on the source code for the paper: [Jin, Di, et al. "Is BERT Really Robust? Natural Language Attack on Text Classification and Entailment." arXiv preprint arXiv:1907.11932 (2019)](https://github.com/jind11/TextFooler).
