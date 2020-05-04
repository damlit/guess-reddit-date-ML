#!/usr/bin/env python
import sys
import pickle
from nltk.tokenize import RegexpTokenizer


# xzcat dev-0/in.tsv.xz | python3 ./predict.py > dev-0/out.tsv


weights, word_to_index_mapping = pickle.load(open('model.pkl', 'rb'))
tokenizer = RegexpTokenizer(r'\b[^\d\W]+\b')

for line in sys.stdin:
    document = line.rstrip()
    terms = tokenizer.tokenize(document)

    y_p = weights[0]
    for term in terms:
        if term in word_to_index_mapping:
            y_p += weights[word_to_index_mapping[term]]

    print(y_p)