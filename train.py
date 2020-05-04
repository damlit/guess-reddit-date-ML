#!/usr/bin/env python
import sys
import random
import pickle

from nltk.tokenize import RegexpTokenizer

# xzcat in.tsv.xz | paste expected.tsv - | ./train.py

amount_of_steps_to_check_error = 10_000
minimum_steps_limit = amount_of_steps_to_check_error * 10
learning_rate = 0.00002
tokenizer = RegexpTokenizer(r'\b[^\d\W]+\b')
empty = "empty"

little_meaning_words_counter = 0
word_index = 1
vocabulary = set()
word_to_index_mapping = {}
words_counter = {}

lines = sys.stdin.readlines()


def fill_vocabulary_and_map(term):
    global word_index
    if term not in vocabulary:
        vocabulary.add(term)
        word_to_index_mapping[term] = word_index
        word_index += 1
    if term in words_counter:
        words_counter[term] += 1
    if term not in words_counter:
        words_counter[term] = 1


def remove_little_meaning_terms_from_vocabulary(word):
    global little_meaning_words_counter
    if words_counter[word] < 5:
        vocabulary.remove(word)
        little_meaning_words_counter += 1


def avg(lines_arg):
    years_summary = 0
    for line in lines_arg:
        line = line.rstrip()
        line = line.split("\t")
        years_summary += float(line[0])
    return years_summary/len(lines)


def is_end_of_train(higher_loss_in_range, steps):
    return higher_loss_in_range > 25 and steps > minimum_steps_limit


def write_model(weights):
    model = (weights, word_to_index_mapping)
    with open('./model.pkl', 'wb') as pickle_file:
        pickle.dump(model, pickle_file)
    print("Model has been saved as model.pkl")


def train():
    empty_text_counter = 0
    for line in lines:
        line = line.rstrip().split("\t")

        if len(line) < 2:
            empty_text_counter += 1
            line.append(empty)

        document = line[1]
        terms = tokenizer.tokenize(document)

        for term in terms:
            fill_vocabulary_and_map(term)

    different_terms_length = len(vocabulary)
    for word in words_counter:
        remove_little_meaning_terms_from_vocabulary(word)

    vocabulary.remove(empty)
    print("--> amount of words with a little meaning: ", little_meaning_words_counter)
    print("--> amount of years without text: ", empty_text_counter)

    weights = [avg(lines)]
    for word_index in range(0, different_terms_length):
        weights.append(random.uniform(-0.01, 0.01))

    steps = 0
    loss_sum = 0
    loss_counter = 0
    lowest_loss = 1_000_000
    higher_loss_in_range = 0
    flag = True

    while flag:
        random_number = random.randrange(0, len(lines))
        line = lines[random_number]
        line = line.rstrip().split("\t")

        if len(line) < 2:
            line.append(empty)

        y = float(line[0].strip())
        document = line[1]
        terms = tokenizer.tokenize(document)

        y_hat = weights[0]
        for term in terms:
            if term in vocabulary:
                y_hat += weights[word_to_index_mapping[term]]

        loss = (y_hat - y) ** 2.0
        loss_sum += loss
        loss_counter += 1

        if loss_counter % amount_of_steps_to_check_error == 0:
            current_loss = loss_sum/loss_counter
            print("current:", current_loss)
            if current_loss > lowest_loss:
                higher_loss_in_range += 1
            else:
                higher_loss_in_range = 0

            if is_end_of_train(higher_loss_in_range, steps):
                flag = False
            if current_loss < lowest_loss:
                lowest_loss = current_loss

            loss_sum = 0
            loss_counter = 0

        delta = (y_hat - y) * learning_rate
        weights[0] = weights[0] - delta

        for term in terms:
            if term in vocabulary:
                weights[word_to_index_mapping[term]] -= delta

        steps += 1

    write_model(weights)


if __name__ == '__main__':
    train()