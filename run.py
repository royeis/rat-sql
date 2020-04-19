import numpy as np
from nltk.stem import WordNetLemmatizer
import torch
import os
from DatasetReader import DatasetReader
from Encoder import Encoder
from RelationAwareMultiheadAttention import RelationAwareMultiheadAttention

lemmatizer = WordNetLemmatizer()


def get_ngrams(tokens, max_n=5):
    ngrams = []
    for i in range(max_n, 0, -1):
        for j in range(len(tokens) - i + 1):
            ngrams.append((' '.join(tokens[j: j + i]), j, j + i))
    return ngrams


class InputExample(object):

    def __init__(self, sequence, relations):
        self.sequence = sequence
        self.relations = relations


class InputLinker(object):

    def __init__(self, question, schema):
        self.q_tokens = question.tokens
        self.q_tokens_relations = np.array(question.q_relations)
        self.column_names = schema.column_names[1:]
        self.table_names = schema.table_names
        self.schema_graph = np.array(schema.graph)
        self.n_q_tokens = len(self.q_tokens)
        self.n_columns = len(self.column_names)
        self.n_tables = len(self.table_names)
        self.n_schema_items = self.n_columns + self.n_tables
        self.relations = self.get_relations_tensor()
        self.schema_embedding = schema.schema_embedding
        self.question_embedding = question.q_embedding
        self.sequence = torch.cat([self.schema_embedding, self.question_embedding], 1)

    def get_example(self):
        return InputExample(self.sequence, self.relations)

    def get_relations_tensor(self):
        q_s_relations, s_q_relations = self.get_schema_linking_relations()
        relations = np.block([[self.schema_graph, s_q_relations], [q_s_relations, self.q_tokens_relations]])
        return torch.tensor(relations).unsqueeze(0)

    def get_schema_linking_relations(self):
        # indicators for tokens that are assigned with type
        tokens_linked = [0 for i in range(len(self.q_tokens))]

        # initialize all relations as:
        # QUESTION-COLUMN-NOMATCH, QUESTION-TABLE-NOMATCH, COLUMN-QUESTION-NOMATCH OR TABLE-QUESTION-NOMATCH
        question_schema_relations = \
            [[21 if i < self.n_columns else 22 for i in range(self.n_schema_items)] for j in range(len(self.q_tokens))]
        schema_question_relations = \
            [[23 if j < self.n_columns else 24 for i in range(len(self.q_tokens))] for j in range(self.n_schema_items)]

        q_ngrams = get_ngrams(self.q_tokens)
        for ngram in q_ngrams:
            # check if ngram matches some column name textually.
            # in that case assign suitable relation of all token with that column,
            # and no match relation with all other schema items
            for i, col in enumerate(self.column_names):
                if sum(tokens_linked[ngram[1]: ngram[2]]) > 0:
                    break
                if ngram[0] == col[1] or lemmatizer.lemmatize(ngram[0]) == col[1]:
                    for j in range(ngram[1], ngram[2]):
                        tokens_linked[j] = 1
                        # QUESTION-COLUMN-EXACT-MATCH
                        question_schema_relations[j][i] = 25
                        # COLUMN-QUESTION-EXACT-MATCH
                        schema_question_relations[i][j] = 26

            for i, col in enumerate(self.column_names):
                if sum(tokens_linked[ngram[1]: ngram[2]]) > 0:
                    break
                if ngram[0] in col[1] or lemmatizer.lemmatize(ngram[0]) in col[1]:
                    for j in range(ngram[1], ngram[2]):
                        tokens_linked[j] = 1
                        # QUESTION-COLUMN-PARTIAL-MATCH
                        question_schema_relations[j][i] = 27
                        # COLUMN-QUESTION-PARTIAL-MATCH
                        schema_question_relations[i][j] = 28

            for i, table in enumerate(self.table_names):
                if sum(tokens_linked[ngram[1]: ngram[2]]) > 0:
                    break
                if ngram[0] == table or lemmatizer.lemmatize(ngram[0]) == table:
                    for j in range(ngram[1], ngram[2]):
                        tokens_linked[j] = 1
                        # QUESTION-TABLE-EXACT-MATCH
                        question_schema_relations[j][i + self.n_columns] = 29
                        # TABLE-QUESTION-EXACT-MATCH
                        schema_question_relations[i + self.n_columns][j] = 30

            for i, table in enumerate(self.table_names):
                if sum(tokens_linked[ngram[1]: ngram[2]]) > 0:
                    break
                if ngram[0] in table or lemmatizer.lemmatize(ngram[0]) in table:
                    for j in range(ngram[1], ngram[2]):
                        tokens_linked[j] = 1
                        # QUESTION-TABLE-PARTIAL-MATCH
                        question_schema_relations[j][i + self.n_columns] = 31
                        # TABLE-QUESTION-PARTIAL-MATCH
                        schema_question_relations[i + self.n_columns][j] = 32

        return np.array(question_schema_relations), np.array(schema_question_relations)


if __name__ == '__main__':
    if not os.path.exists('processed_input.pt'):
        print('reading and processing data')
        dr = DatasetReader('spider/')
        dr.read_all()
        input_examples = []
        for i, q in enumerate(dr.processed_questions):
            if i % 100 == 0:
                print(f'example {i} / {len(dr.processed_questions)}')
            s = dr.schema_dict[q.db_id]
            input_examples.append(InputLinker(q, s).get_example())
        torch.save(input_examples, 'processed_input.pt')
    else:
        print('using processed data from cache')
        input_examples = torch.load('processed_input.pt')

    # test
    encoder = Encoder()
    logits = encoder(input_examples[0].sequence, input_examples[0].relations)

    print('done')