from typing import List, Tuple, Dict, Set, Optional
from copy import deepcopy
from SpiderContext import Question, Schema
from Grammar import GRAMMAR_DICTIONARY, update_grammar_to_be_table_names_free, update_grammar_with_tables
from Utils import format_grammar_string, initialize_valid_actions, SqlVisitor
from parsimonious import Grammar
from parsimonious.exceptions import ParseError
from allennlp.data.fields import KnowledgeGraphField


class SpiderWorld:
    def __init__(self, question: Question, schema: Schema, query: Optional[List[str]]):
        self.question = question
        self.schema = schema
        self.query = query
        # TODO probably change entities_names
        self.entities_names = [col[1] for col in schema.column_names_original if col[1] != '*']
        self.entities_names += schema.table_names_original
        self.base_grammar_dictionary = deepcopy(GRAMMAR_DICTIONARY)

        self.valid_actions = []
        self.valid_actions_flat = []

    def get_action_sequence_and_all_actions(self) -> Tuple[List[str], List[str]]:
        grammar_with_context = deepcopy(self.base_grammar_dictionary)

        update_grammar_to_be_table_names_free(grammar_with_context)
        schema = self.schema

        update_grammar_with_tables(grammar_with_context, schema)
        grammar = Grammar(format_grammar_string(grammar_with_context))

        valid_actions = initialize_valid_actions(grammar)
        all_actions = set()
        for action_list in valid_actions.values():
            all_actions.update(action_list)
        sorted_actions = sorted(all_actions)
        self.valid_actions = valid_actions
        self.valid_actions_flat = sorted_actions

        action_sequence = None
        if self.query is not None:
            sql_visitor = SqlVisitor(grammar)
            query = " ".join(self.query).lower().replace("``", "'").replace("''", "'")
            try:
                action_sequence = sql_visitor.parse(query) if query else []
            except ParseError as e:
                pass

        return action_sequence, sorted_actions

    def is_global_rule(self, rhs: str) -> bool:
        rhs = rhs.strip('[] ')
        if rhs[0] != '"':
            return True
        return rhs.strip('"') not in self.entities_names
