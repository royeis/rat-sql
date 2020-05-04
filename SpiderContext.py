import torch
from transformers import BertModel, BertTokenizer
from Utils import disambiguate_items

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer.bos_token = '[CLS]'
tokenizer.eos_token = '[SEP]'
encoder = BertModel.from_pretrained('bert-base-uncased')
encoder.eval()


def clip(a, d=2):
    return max(-1 * d, min(d, a))


class Question(object):

    def __init__(self, question_dict, schema):
        self.db_id = question_dict['db_id']
        self.tokens = [tok.lower() for tok in question_dict['question_toks']]
        self.query_toks = question_dict['query_toks_no_value'] or None
        if self.query_toks is not None:
            self.query_toks = disambiguate_items(self.db_id, self.query_toks, schema)
        self.sql = question_dict['sql']
        self.q_relations = self.prepare_question_relations()
        self.q_embedding = self.get_question_embedding()

    def prepare_question_relations(self):
        relations = [[18 for i in range(len(self.tokens))] for j in range(len(self.tokens))]
        for i in range(len(self.tokens) - 1):
            for j in range(i, len(self.tokens)):
                d = clip(j-i)
                relations[i][j] += d
                relations[j][i] -= d
        return relations

    def get_question_embedding(self):
        tokens = [tokenizer.bos_token] + self.tokens + [tokenizer.eos_token]
        q_tokenizer_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).unsqueeze(0)

        with torch.no_grad():
            embeds = encoder(q_tokenizer_ids)[0]
            return embeds.narrow(1, 1, len(self.tokens))


class Schema(object):

    def __init__(self, schema_dict):
        self.db_id = schema_dict['db_id']
        self.table_names_original = schema_dict['table_names_original']
        self.column_names_original = schema_dict['column_names_original']
        self.table_names = schema_dict['table_names']
        self.column_names = schema_dict['column_names']
        self.column_types = schema_dict['column_types']
        self.primary_keys = schema_dict['primary_keys']
        self.foreign_keys = schema_dict['foreign_keys']
        self.graph = self.prepare_schema_graph()
        self.schema_embedding = self.get_schema_embedding()

    def prepare_schema_graph(self):
        # We represent the schema graph as a 2D array of relations.
        # The first row and column of the graph representation refer to a dummy column in the column list,
        # that is part of the dataset format.
        # All types of edges are based on RAT-SQL article.

        n_cols = len(self.column_names)
        n_tables = len(self.table_names)
        n = n_cols + n_tables

        # initialize graph without edges.
        graph = [[-1 for i in range(n)] for j in range(n)]

        # set all column-column edges
        for i in range(1, n_cols):
            for j in range(1, n_cols):
                # COLUMN-IDENTITY
                if i == j:
                    graph[i][j] = 0

                # SAME-TABLE
                elif self.column_names[i][0] == self.column_names[j][0]:
                    graph[i][j] = 1

                # FOREIGN-KEY-COL-F, FOREIGN-KEY-COL-R
                elif [i, j] in self.foreign_keys:
                    graph[i][j] = 2
                    graph[j][i] = 3

                # COLUMN-COLUMN
                else:
                    graph[i][j] = 4

        # set all column-table and table-column edges
        # PRIMARY-KEY-F, PRIMARY-KEY-R
        for k in self.primary_keys:
            graph[k][n_cols + self.column_names[k][0]] = 5
            graph[n_cols + self.column_names[k][0]][k] = 6

        for i in range(1, n_cols):
            for j in range(n_cols, n):
                # check if type already assigned
                if graph[i][j] != -1:
                    continue

                # BELONGS-TO-F, BELONGS-TO-R
                elif self.column_names[i][0] == j - n_cols:
                    graph[i][j] = 7
                    graph[j][i] = 8

                # COLUMN-TABLE, TABLE-COLUMN
                else:
                    graph[i][j] = 9
                    graph[j][i] = 10

        # set all table-table edges
        for fk in self.foreign_keys:
            t1 = self.column_names[fk[0]][0]
            t2 = self.column_names[fk[1]][0]

            # FOREIGN-KEY-TAB-F, FOREIGN-KEY-TAB-R
            if graph[n_cols + t1][n_cols + t2] == -1:
                graph[n_cols + t1][n_cols + t2] = 11
                graph[n_cols + t2][n_cols + t1] = 12

            # FOREIGN-KEY-B
            else:
                graph[n_cols + t1][n_cols + t2] = graph[n_cols + t2][n_cols + t2] = 13

        for i in range(n_cols, n):
            for j in range(n_cols, n):
                # TABLE-IDENTITY
                if i == j:
                    graph[i][j] = 14

                # TABLE-TABLE
                elif graph[i][j] == -1:
                    graph[i][j] = 15

        # clip the first row and column that refer to the dummy column
        graph = [graph[i][1:] for i in range(len(graph))][1:]
        return graph

    def get_schema_embedding(self):
        col_tokenizer_ids = [torch.tensor(tokenizer.encode(col[1])).unsqueeze(0) for col in self.column_names[1:]]
        table_tokenizer_ids = [torch.tensor(tokenizer.encode(table)).unsqueeze(0) for table in self.table_names]

        with torch.no_grad():
            cols_embeds = [encoder(ids)[1].unsqueeze(1) for ids in col_tokenizer_ids]
            tables_embeds = [encoder(ids)[1].unsqueeze(1) for ids in table_tokenizer_ids]

        schema_embeds = cols_embeds + tables_embeds
        return torch.cat(schema_embeds, 1)


class SpiderContext(object):
    def __init__(self, question: Question, schema: Schema):
        self.question = question
        self.schema = schema



