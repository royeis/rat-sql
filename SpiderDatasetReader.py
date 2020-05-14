import json
import numpy as np
import torch
from nltk.stem import WordNetLemmatizer
from SpiderWorld import SpiderWorld
from SpiderContext import SpiderContext, Question, Schema
import os

lemmatizer = WordNetLemmatizer()


def get_ngrams(tokens, max_n=5):
    ngrams = []
    for i in range(max_n, 0, -1):
        for j in range(len(tokens) - i + 1):
            ngrams.append((' '.join(tokens[j: j + i]), j, j + i))
    return ngrams


class SpiderDatasetReader(object):

    def __init__(self):
        super().__init__(lazy=False)
        self.questions = []
        self.schemas = []
        self.schema_dict = {}
        self.processed_questions = []

    # Since there are multiple input files file_path we assume file_path points to the dataset directory
    def _read(self, file_path):
        self.dataset_path = file_path

        if not os.path.exists('processed_input.pt'):
            print('reading and processing data')
            self.read_schemas()
            self.read_questions()
            input_instances = []
            for i, q in enumerate(self.processed_questions):
                if i % 100 == 0:
                    print(f'example {i} / {len(self.processed_questions)}')
                s = self.schema_dict[q.db_id]
                context = SpiderContext(q, s)
                input_instances.append(self.text_to_instance(context))
            torch.save(input_instances, 'processed_input.pt')
        else:
            print('using processed data from cache')
            input_instances = torch.load('processed_input.pt')
        return input_instances

    def text_to_instance(self, context: SpiderContext):
        instance = {}
        question = context.question
        schema = context.schema

        q_tokens = question.tokens
        q_tokens_relations = np.array(question.q_relations)
        column_names = schema.column_names[1:]
        table_names = schema.table_names
        schema_graph = np.array(schema.graph)
        relations = self.get_relations_tensor(schema_graph, q_tokens_relations, q_tokens, column_names, table_names)
        schema_embedding = schema.schema_embedding
        question_embedding = question.q_embedding
        sequence = torch.cat([schema_embedding, question_embedding], 1)

        instance['sequence'] = sequence
        instance['relations'] = relations

        world = SpiderWorld(question, schema, query=question.query_toks)

        action_sequence, all_actions = world.get_action_sequence_and_all_actions()

        if action_sequence is None:
            print("Parse error")
            action_sequence = []
        elif action_sequence is None:
            return None

        action_sequence_indices = []
        valid_actions = []

        for production_rule in all_actions:
            nonterminal, rhs = production_rule.split(' -> ')
            production_rule = ' '.join(production_rule.split(' '))

            prod_dict = {'rule': production_rule, 'global': world.is_global_rule(rhs), 'nonterminal': nonterminal}

            valid_actions.append(prod_dict)

        instance["valid_actions"] = valid_actions

        action_map = {action['rule']: i  # type: ignore
                      for i, action in enumerate(valid_actions)}

        for production_rule in action_sequence:
            action_sequence_indices.append(action_map[production_rule])
        if not action_sequence:
            action_sequence_indices = []

        instance["action_sequence"] = action_sequence_indices
        instance["world"] = world
        return instance

    def get_relations_tensor(self,
                             schema_graph,
                             q_tokens_relations,
                             q_tokens,
                             column_names,
                             table_names):
        q_s_relations, s_q_relations = self.get_schema_linking_relations(q_tokens, column_names, table_names)
        relations = np.block([[schema_graph, s_q_relations], [q_s_relations, q_tokens_relations]])
        return torch.tensor(relations).unsqueeze(0)

    def get_schema_linking_relations(self, q_tokens, column_names, table_names):
        n_q_tokens = len(q_tokens)
        n_columns = len(column_names)
        n_tables = len(table_names)
        n_schema_items = n_columns + n_tables

        # indicators for tokens that are assigned with type
        tokens_linked = [0 for i in range(n_q_tokens)]

        # initialize all relations as:
        # QUESTION-COLUMN-NOMATCH, QUESTION-TABLE-NOMATCH, COLUMN-QUESTION-NOMATCH OR TABLE-QUESTION-NOMATCH
        question_schema_relations = \
            [[21 if i < n_columns else 22 for i in range(n_schema_items)] for j in range(n_q_tokens)]
        schema_question_relations = \
            [[23 if j < n_columns else 24 for i in range(n_q_tokens)] for j in range(n_schema_items)]

        q_ngrams = get_ngrams(q_tokens)
        for ngram in q_ngrams:
            # check if ngram matches some column name textually.
            # in that case assign suitable relation of all token with that column,
            # and no match relation with all other schema items
            for i, col in enumerate(column_names):
                if sum(tokens_linked[ngram[1]: ngram[2]]) > 0:
                    break
                if ngram[0] == col[1] or lemmatizer.lemmatize(ngram[0]) == col[1]:
                    for j in range(ngram[1], ngram[2]):
                        tokens_linked[j] = 1
                        # QUESTION-COLUMN-EXACT-MATCH
                        question_schema_relations[j][i] = 25
                        # COLUMN-QUESTION-EXACT-MATCH
                        schema_question_relations[i][j] = 26

            for i, col in enumerate(column_names):
                if sum(tokens_linked[ngram[1]: ngram[2]]) > 0:
                    break
                if ngram[0] in col[1] or lemmatizer.lemmatize(ngram[0]) in col[1]:
                    for j in range(ngram[1], ngram[2]):
                        tokens_linked[j] = 1
                        # QUESTION-COLUMN-PARTIAL-MATCH
                        question_schema_relations[j][i] = 27
                        # COLUMN-QUESTION-PARTIAL-MATCH
                        schema_question_relations[i][j] = 28

            for i, table in enumerate(table_names):
                if sum(tokens_linked[ngram[1]: ngram[2]]) > 0:
                    break
                if ngram[0] == table or lemmatizer.lemmatize(ngram[0]) == table:
                    for j in range(ngram[1], ngram[2]):
                        tokens_linked[j] = 1
                        # QUESTION-TABLE-EXACT-MATCH
                        question_schema_relations[j][i + n_columns] = 29
                        # TABLE-QUESTION-EXACT-MATCH
                        schema_question_relations[i + n_columns][j] = 30

            for i, table in enumerate(table_names):
                if sum(tokens_linked[ngram[1]: ngram[2]]) > 0:
                    break
                if ngram[0] in table or lemmatizer.lemmatize(ngram[0]) in table:
                    for j in range(ngram[1], ngram[2]):
                        tokens_linked[j] = 1
                        # QUESTION-TABLE-PARTIAL-MATCH
                        question_schema_relations[j][i + n_columns] = 31
                        # TABLE-QUESTION-PARTIAL-MATCH
                        schema_question_relations[i + n_columns][j] = 32

        return np.array(question_schema_relations), np.array(schema_question_relations)

    def read_schemas(self, schemas_files=('tables.json', )):
        for f in schemas_files:
            self.read_schemas_file(f)
        self.prepare_schema_dict()

    def read_questions(self, questions_files=('train_spider.json', 'train_others.json') ):
        for f in questions_files:
            self.read_questions_file(f)
        self.process_questions()

    def read_schemas_file(self, file_name):
        with open(self.dataset_path + file_name) as f:
            schemas_read = json.load(f)
        self.schemas += schemas_read

    def read_questions_file(self, file_name):
        with open(self.dataset_path + file_name) as f:
            questions_read = json.load(f)
        self.questions += questions_read

    def process_questions(self):
        print('processing questions')
        for i, q in enumerate(self.questions[:100]):
            if i % 200 == 0:
                print(f'question: {i} / {len(self.questions) - 1}')
            self.processed_questions.append(Question(q, self.schema_dict[q['db_id']]))

    def prepare_schema_dict(self):
        print('processing schemas')
        for i, s in enumerate(self.schemas):
            if i % 10 == 0:
                print(f'schema: {i} / {len(self.schemas) - 1}')
            self.schema_dict[s['db_id']] = Schema(s)


if __name__ == '__main__':

    dr = SpiderDatasetReader()
    instances = dr._read('spider/')
    for ins in instances:
        print(ins)
    print('done')
