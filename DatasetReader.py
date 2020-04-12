import json

class DatasetReader(object):

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.schemas = []
        self.questions = []

    def read_schemas_file(self, file_name):
        with open(self.dataset_path + file_name) as f:
            schemas_read = json.load(f)
        self.schemas += schemas_read

    def read_questions_file(self, file_name):
        with open(self.dataset_path + file_name) as f:
            questions_read = json.load(f)
        self.questions += questions_read


class Schema(object):

    def __init__(self, schema_dict):
        raise NotImplementedError()


class Question(object):

    def __init__(self, question_dict):
        self.db_id = question_dict['db_id']
        self.tokens = question_dict['question_toks']


if __name__ == '__main__':
    dr = DatasetReader('spider/')
    dr.read_schemas_file('tables.json')
    dr.read_questions_file('train_spider.json')
    dr.read_questions_file('train_others.json')

    schemas = [Schema(s) for s in dr.schemas]
    questions = [Question(q) for q in dr.questions]




    print('done')