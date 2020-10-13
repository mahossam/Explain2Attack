#%%

from sklearn.externals.joblib import Parallel, delayed
# from math import sqrt
import time


def something(n):
    print(f"{n}")
    time.sleep(1.0)
    return n + 1


if __name__ == '__main__':
    Parallel(n_jobs=4)(delayed(something)(i ** 2) for i in range(10))

#%%
import numpy as np
l2x_words = np.load(r'/home/mahmoudm/pb90/mahmoud/TextFooler-master/adv_results_1585219499/word_ranking_bert_l2x.npy', allow_pickle=True)
jin_words = np.load(r'/home/mahmoudm/pb90/mahmoud/TextFooler-master/adv_results_1585219582/word_ranking_bert_jin.npy', allow_pickle=True)

l2x_final_words = np.array([words if len(words) > 0 else [-1]*20 for words in l2x_words])
jin_final_words = np.array([words if len(words) > 0 else [-1]*20 for words in jin_words])

l2x_total_count = np.sum([len(sent) for sent in l2x_final_words])
jin_total_count = np.sum([len(sent) for sent in jin_final_words])
jin_matched_count = 0
for i, sent in enumerate(jin_final_words):
    for word in sent:
        if word in l2x_final_words[i]:
            jin_matched_count += 1
jin_matched_percentage = jin_matched_count / l2x_total_count
