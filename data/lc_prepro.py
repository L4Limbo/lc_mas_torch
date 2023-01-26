import os
import json
import h5py
import copy
import argparse
import numpy as np
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download('punkt')
nltk.download('stopwords')
stop_words = stopwords.words('english')

MAX_QUESTION_LEN = 40
MAX_ANSWER_LEN = 40
MAX_DOCUMENT_LEN = 1400
MAX_SUMMARY_LEN = 200
NUMBER_OF_ROUNDS = 10

SPLIT_INDEX = 117

DOC_PATH = './generated_data/gen_dataset.json'
SUMM_PATH = './generated_data/summary_dataset.json'
INPUT_PATH = './generated_data/gen_dataset.json'
TRAIN_PATH = './generated_data/train.json'
VAL_PATH = './generated_data/val.json'

# Output Files
OUTPUT_JSON = './processed_data/processed_data.json'
OUTPUT_H5 = './processed_data/processed_data.h5'
OUTPUT_SUMM = './processed_data/summ.json'


def tokenize_data(data, word_count=False):
    '''
    Tokenize documents, questions and answers
    Also maintain word count if required
    '''
    res, word_counts = {}, {}

    print('Tokenizing data and documents...')
    for i in data['data']['dialogs']:
        summary_id = i['summary']
        words = nltk.word_tokenize(i['document'])
        words = [word.lower() for word in words if word.isalnum()]
        document = word_tokenize(' '.join(words))
        clear_document = []
        for word in document:
            if (word not in stop_words):
                clear_document.append(word)
        res[summary_id] = {'document': clear_document}

    print('Tokenizing questions...')
    ques_toks, ans_toks = [], []
    for i in data['data']['questions']:
        words = nltk.word_tokenize(i)
        words = [word.lower() for word in words if word.isalnum()]
        ques_toks.append(word_tokenize(' '.join(words) + '?'))
        clear_ques = []
        for word in ques_toks:
            if (word not in stop_words):
                clear_ques.append(word)
                
    print('Tokenizing answers...')
    for i in data['data']['answers']:
        words = nltk.word_tokenize(i)
        words = [word.lower() for word in words if word.isalnum()]
        ans_toks.append(word_tokenize(' '.join(words)))
        clear_ans = []
        for word in ans_toks:
            if (word not in stop_words):
                clear_ans.append(word)

    for i in data['data']['dialogs']:
        # last round of dialog will not have answer for test split
        if 'answer' not in i['dialog'][-1]:
            i['dialog'][-1]['answer'] = -1
        res[i['summary']]['num_rounds'] = len(i['dialog'])
        # right-pad i['dialog'] with empty question-answer pairs at the end
        while len(i['dialog']) < NUMBER_OF_ROUNDS:
            i['dialog'].append({'question': random.choice(range(
                len(data['data']['questions']))), 'answer': random.choice(range(len(data['data']['answers'])))})
        res[i['summary']]['dialog'] = i['dialog']
        if word_count == True:
            for j in range(NUMBER_OF_ROUNDS):
                question = clear_ques[i['dialog'][j]['question']]
                answer = clear_ans[i['dialog'][j]['answer']]
                for word in question + answer:
                    word_counts[word] = word_counts.get(word, 0) + 1

    return res, clear_ques, clear_ans, word_counts


def encode_summaries(data_toks, word2ind):
    res = {}
    for key in data_toks.keys():
        res[key] = [word2ind.get(word, word2ind['UNK'])
                    for word in data_toks[key]][:MAX_SUMMARY_LEN]
    return res


def encode_vocab(data_toks, ques_toks, ans_toks, word2ind):
    '''
    Converts string tokens to indices based on given dictionary
    '''
    max_ques_len, max_ans_len, max_doc_len = 0, 0, 0
    for k, v in data_toks.items():
        summary_id = k
        document = [word2ind.get(word, word2ind['UNK'])
                    for word in v['document']]
        if max_doc_len < len(document):
            max_doc_len = len(document)
        data_toks[k]['document_inds'] = document
        data_toks[k]['document_len'] = len(document)

    ques_inds, ans_inds = [], []
    for i in ques_toks:
        question = [word2ind.get(word, word2ind['UNK'])
                    for word in i]
        ques_inds.append(question)

    for i in ans_toks:
        answer = [word2ind.get(word, word2ind['UNK'])
                  for word in i]
        ans_inds.append(answer)

    return data_toks, ques_inds, ans_inds


def split_data(json_data, train_path, val_path):
    train_data, val_data, test_data = {
        'data': {}}, {'data': {}}, {'data': {}}

    data_size = len(json_data['data']['dialogs'])

    train_data['data']['questions'] = json_data['data']['questions'][:]
    val_data['data']['questions'] = json_data['data']['questions'][:]
    test_data['data']['questions'] = json_data['data']['questions'][:]

    train_data['data']['answers'] = json_data['data']['answers'][:]
    val_data['data']['answers'] = json_data['data']['answers'][:]
    test_data['data']['answers'] = json_data['data']['answers'][:]

    train_data['data']['dialogs'] = json_data['data']['dialogs'][0:SPLIT_INDEX]
    val_data['data']['dialogs'] = json_data['data']['dialogs'][SPLIT_INDEX:]

    with open(train_path, 'w') as jsonFile:
        jsonFile.write(json.dumps(train_data, indent=4))

    with open(val_path, 'w') as jsonFile:
        jsonFile.write(json.dumps(val_data, indent=4))

    return train_data, val_data


def create_data_mats(data_toks, ques_inds, ans_inds, dtype):
    num_threads = len(data_toks.keys())

    print('Creating data mats for %s...' % dtype)

    # create summary lists and document data mats
    summary_list = []
    summary_index = np.zeros(num_threads)
    max_doc_len = MAX_DOCUMENT_LEN
    documents = np.zeros([num_threads, max_doc_len])
    document_len = np.zeros(num_threads, dtype=np.int64)
    summary_ids = list(data_toks.keys())

    for i in range(num_threads):
        summary_id = summary_ids[i]
        if dtype == 'test':
            summary_list.append('%s' %
                                (summary_id))
        else:
            summary_list.append('%s' %
                                (summary_id))
        summary_index[i] = i
        document_len[i] = len(data_toks[summary_id]
                              ['document_inds'][0:max_doc_len])
        documents[i][0:document_len[i]
                     ] = data_toks[summary_id]['document_inds'][0:max_doc_len]

    num_rounds = NUMBER_OF_ROUNDS
    max_ques_len = MAX_ANSWER_LEN
    max_ans_len = MAX_QUESTION_LEN

    questions = np.zeros([num_threads, num_rounds, max_ques_len])
    answers = np.zeros([num_threads, num_rounds, max_ans_len])
    question_len = np.zeros([num_threads, num_rounds], dtype=np.int64)
    answer_len = np.zeros([num_threads, num_rounds], dtype=np.int64)

    # create questions and answers data mats
    for i in range(num_threads):
        summary_id = summary_ids[i]
        for j in range(num_rounds):
            if data_toks[summary_id]['dialog'][j]['question'] != -1:
                question_len[i][j] = len(
                    ques_inds[data_toks[summary_id]['dialog'][j]['question']][0:max_ques_len])
                questions[i][j][0:question_len[i][j]] = ques_inds[data_toks[summary_id]
                                                                  ['dialog'][j]['question']][0:max_ques_len]
            if data_toks[summary_id]['dialog'][j]['answer'] != -1:
                answer_len[i][j] = len(
                    ans_inds[data_toks[summary_id]['dialog'][j]['answer']][0:max_ans_len])
                answers[i][j][0:answer_len[i][j]] = ans_inds[data_toks[summary_id]
                                                             ['dialog'][j]['answer']][0:max_ans_len]

    # create ground truth answer and options data mats
    answer_index = np.zeros([num_threads, num_rounds])
    num_rounds_list = np.full(num_threads, NUMBER_OF_ROUNDS)
    options = np.zeros([num_threads, num_rounds, 100])

    for i in range(num_threads):
        summary_id = summary_ids[i]
        for j in range(num_rounds):
            if (data_toks[summary_id]['dialog'][j]['answer'] != -1):
                rand_padding = np.array([random.choice(range(100))
                                        for i in range(100)])

                data_toks[summary_id]['dialog'][j]['answer_options'] = [
                    random.choice(range(100)) for i in range(100)]
                data_toks[summary_id]['dialog'][j]['gt_index'] = random.choice(
                    range(100))
                options[i][j] = np.concatenate((np.array(
                    data_toks[summary_id]['dialog'][j]['answer_options'][:100]), rand_padding[:100-len(data_toks[summary_id]['dialog'][j]['answer_options'][:100])])) + 1
                answer_index[i][j] = data_toks[summary_id]['dialog'][j]['gt_index'] + 1
                
    options_list = np.zeros([len(ans_inds), max_ans_len])
    options_len = np.zeros(len(ans_inds), dtype=np.int64)

    for i in range(len(ans_inds)):
        options_len[i] = len(ans_inds[i][0:max_ans_len])
        options_list[i][0:options_len[i]] = ans_inds[i][0:max_ans_len]

    return documents, document_len, questions, question_len, answers, answer_len, options, options_list, options_len, answer_index, summary_index, summary_list, num_rounds_list


def tokenize_summaries():
    summ_path = SUMM_PATH
    json_data = json.load(open(summ_path))
    res, word_counts_summ = {}, {}

    for key in json_data.keys():
        words = nltk.word_tokenize(json_data[key])

        words = [word.lower() for word in words if word.isalnum()]

        res[key] = word_tokenize(' '.join(words))
        clear_summ = []
        for word in res[key]:
            if (word not in stop_words):
                clear_summ.append(word)
        res[key] = clear_summ

        # while len(res[key]) < MAX_SUMMARY_LEN:
        #     res[key].append(' ')

    for key in res.keys():
        for word in res[key]:
            word_counts_summ[word] = word_counts_summ.get(word, 0) + 1

    return res, word_counts_summ


def tokenize_documents():
    doc_path = DOC_PATH
    json_data = json.load(open(doc_path))
    res, word_counts_docs = {}, {}

    for dialog in json_data['data']['dialogs']:
        words = nltk.word_tokenize(dialog['document'])
        words = [word.lower() for word in words if word.isalnum()]
        res[dialog['summary']] = word_tokenize(' '.join(words))
        clear_docs = []
        for word in res[dialog['summary']]:
            if (word not in stop_words):
                clear_docs.append(word)
        res[dialog['summary']] = clear_docs

    for key in res.keys():
        for word in res[key]:
            word_counts_docs[word] = word_counts_docs.get(word, 0) + 1

    return res, word_counts_docs


if __name__ == "__main__":

    print('Preprocessing ...')
    # Input Files
    input_path = INPUT_PATH
    train_path = TRAIN_PATH
    val_path = VAL_PATH

    # Output Files
    output_json = OUTPUT_JSON
    output_h5 = OUTPUT_H5
    output_summ = OUTPUT_SUMM

    os.makedirs('./processed_data',  exist_ok=True)

    # Load data
    print('Loading Data ...')
    json_data = json.load(open(input_path))

    # Split Dataset 75 / 25
    print('Creating Splits ...')
    data_train, data_val = split_data(
        json_data, train_path, val_path)

    # Tokenizing
    data_train_toks, ques_train_toks, ans_train_toks, word_counts_train = tokenize_data(
        data_train, True)
    data_val_toks, ques_val_toks, ans_val_toks, word_counts_val = tokenize_data(
        data_val, True)

    print('Parsing and Tokenizing summaries')
    summaries_data_toks, word_counts_summ = tokenize_summaries()

    print('Parsing and Tokenizing documents')
    documents_data_toks, word_counts_docs = tokenize_documents()

    print('Building vocabulary...')

    word_counts_all = dict(word_counts_train)
    word_counts_all.update(dict(word_counts_val))
    word_counts_all.update(dict(word_counts_summ))
    word_counts_all.update(dict(word_counts_docs))

    for word, count in word_counts_val.items():
        word_counts_all[word] = word_counts_all.get(word, 0) + count

    for word, count in word_counts_summ.items():
        word_counts_all[word] = word_counts_all.get(word, 0) + count

    for word, count in word_counts_docs.items():
        word_counts_all[word] = word_counts_all.get(word, 0) + count

    word_counts_all['UNK'] = 5
    vocab = [word for word in word_counts_all
             if word_counts_all[word] >= 5]
    print('Words: %d' % len(vocab))
    word2ind = {word: word_ind+1 for word_ind, word in enumerate(vocab)}
    ind2word = {word_ind: word for word, word_ind in word2ind.items()}

    print('Encoding based on vocabulary...')
    data_train_toks, ques_train_inds, ans_train_inds = encode_vocab(
        data_train_toks, ques_train_toks, ans_train_toks, word2ind)
    data_val_toks, ques_val_inds, ans_val_inds = encode_vocab(
        data_val_toks, ques_val_toks, ans_val_toks, word2ind)

    summaries_data_toks = encode_summaries(summaries_data_toks, word2ind)
    json.dump(summaries_data_toks, open(output_summ, 'w'))

    print('Creating data matrices...')
    documents_train, documents_train_len, questions_train, questions_train_len, answers_train, answers_train_len, options_train, options_train_list, options_train_len, answers_train_index, summarys_train_index, summarys_train_list, _ = create_data_mats(
        data_train_toks, ques_train_inds, ans_train_inds, 'train')
    documents_val, documents_val_len, questions_val, questions_val_len, answers_val, answers_val_len, options_val, options_val_list, options_val_len, answers_val_index, summarys_val_index, summarys_val_list, _ = create_data_mats(
        data_val_toks, ques_val_inds, ans_val_inds, 'val')


    print('Saving hdf5...')
    f = h5py.File(output_h5, 'w')
    f.create_dataset('ques_train', dtype='uint32', data=questions_train)
    f.create_dataset('ques_length_train', dtype='uint32',
                     data=questions_train_len)
    f.create_dataset('ans_train', dtype='uint32', data=answers_train)
    f.create_dataset('ans_length_train', dtype='uint32',
                     data=answers_train_len)
    f.create_dataset('ans_index_train', dtype='uint32',
                     data=answers_train_index)
    f.create_dataset('doc_train', dtype='uint32', data=documents_train)
    f.create_dataset('doc_length_train', dtype='uint32',
                     data=documents_train_len)
    f.create_dataset('opt_train', dtype='uint32', data=options_train)
    f.create_dataset('opt_length_train', dtype='uint32',
                     data=options_train_len)
    f.create_dataset('opt_list_train', dtype='uint32', data=options_train_list)
    f.create_dataset('summ_pos_train', dtype='uint32',
                     data=summarys_train_index)

    f.create_dataset('ques_val', dtype='uint32', data=questions_val)
    f.create_dataset('ques_length_val', dtype='uint32', data=questions_val_len)
    f.create_dataset('ans_val', dtype='uint32', data=answers_val)
    f.create_dataset('ans_length_val', dtype='uint32', data=answers_val_len)
    f.create_dataset('ans_index_val', dtype='uint32', data=answers_val_index)
    f.create_dataset('doc_val', dtype='uint32', data=documents_val)
    f.create_dataset('doc_length_val', dtype='uint32', data=documents_val_len)
    f.create_dataset('opt_val', dtype='uint32', data=options_val)
    f.create_dataset('opt_length_val', dtype='uint32', data=options_val_len)
    f.create_dataset('opt_list_val', dtype='uint32', data=options_val_list)
    f.create_dataset('summ_pos_val', dtype='uint32', data=summarys_val_index)

    f.close()

    out = {}
    out['ind2word'] = ind2word
    out['word2ind'] = word2ind

    out['unique_summ_train'] = summarys_train_list
    out['unique_summ_val'] = summarys_val_list

    json.dump(out, open(output_json, 'w'))