import json
import h5py
import numpy as np
import string
import re
import nltk
import random
import torch
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from question_generation.pipelines import pipeline


def generate_dataset(gen_pipeline: pipeline, data: json) -> None:

    # Initialize Data Dictionary
    dataset = {}
    summary_dataset = {}
    dataKeys = data.keys()
    dataset['questions'], dataset['answers'], dataset['dialogs'] = [], [], []

    summary_id = 0
    counter = 0

    print('Parsing dataset initiated ...')
    for key in dataKeys:
        counter += 1
        print(counter)
        if (counter > 5):
            break
        # Store questions' and answers' last ids
        questions_id_counter = len(dataset['questions'])
        answers_id_counter = len(dataset['answers'])

        # Append Question
        dataset['questions'].append(data[key]['question'])

        # Append Answers
        answer_keys = list(data[key]['answers'].keys())

        for answer_key in answer_keys:
            dataset['answers'].append(
                data[key]['answers'][answer_key]['answer_ext_summ'])

        # Generate Dialogs from dataset
        current_dialog = {
            'summary': '',
            'document': '',
            'dialog': [],
        }

        # Set summary and save it to summaries' dataset
        summary_dataset[str(summary_id)] = data[key]['multi_ext_summ']
        current_dialog['summary'] = str(summary_id)
        summary_id += 1

        # Set the first article as the dialog's document
        current_dialog['document'] = data[key]['answers'][answer_keys[0]]['article']

        # Generate Dialogues from the standard dataset
        dialogue = {
            'question': str(questions_id_counter),
            'answer': str(answers_id_counter),
            'answer_options': [i for i in range(answers_id_counter, len(dataset['answers']))],
            'gt_index': '',
        }
        dialogue['gt_index'] = '0'

        current_dialog['dialog'].append(dialogue)

        # Generate more questions and answers with transformer pipeline
        for answer_key in answer_keys:
            # try:
            print('trying to generate...')
            # Generate questions and answers for each article in the answers

            generated_qas = gen_pipeline(
                data[key]['answers'][answer_key]['article'])
            # except:
            #     print('error')
            #     continue

            # Append Results in the generated dataset and Create new dialogues
            prev_answer_id_counter = answers_id_counter
            for qa in generated_qas:

                questions_id_counter = len(dataset['questions'])
                answers_id_counter = len(dataset['answers'])

                dataset['questions'].append(qa['question'])
                dataset['answers'].append(qa['answer'])

                # Generate Dialogues from generated question-answers
                print('%s %s' % prev_answer_id_counter %
                      len(dataset['answers']))
                dialogue = {
                    'question': str(len(dataset['questions']) - 1),
                    'answer': str(len(dataset['answers']) - 1),
                    'answer_options': [i for i in range(prev_answer_id_counter, len(dataset['answers']))],
                    'gt_index': '',
                }
                dialogue['gt_index'] = str(len(
                    dialogue['answer_options'] - 1))

                current_dialog['dialog'].append(dialogue)

        dataset['dialogs'].append(current_dialog)

    print('Ready to save dataset ...')
    dataset_to_create = {
        'data': dataset
    }

    with open('./data/generated_data/gen_dataset.json', 'w') as jsonFile:
        jsonFile.write(json.dumps(dataset_to_create, indent=4))

    print('Created generated dataset ...')


def main():
    torch.cuda.memory_summary(device=None, abbreviated=False)
    # Question Driven Answer Summarization Primary Dataset path
    mediqa_ans_summ_dataset_path = './data/raw_data/question_driven_answer_summarization_primary_dataset.json'

    # Load Pipeline for QA Generation
    print('Loading pipeline ...')
    qa_gen_pipeline = pipeline(
        'multitask-qa-qg', model="valhalla/t5-base-qa-qg-hl")

    # Generate Dataset with VisDial-like structure
    print('Loading dataset ...')
    jsonData = json.load(open(mediqa_ans_summ_dataset_path))
    print(qa_gen_pipeline(jsonData['1']['answers']['1_Answer1']['article']))
    # generate_dataset(qa_gen_pipeline, jsonData)

    pass


if __name__ == '__main__':
    main()
