import re
import json
from collections import defaultdict
from sys import exc_info
from typing import List, Dict, Set, Optional

from overrides import overrides
from allennlp.common import JsonDict
from parsimonious.exceptions import VisitationError, UndefinedLabel
from parsimonious.expressions import Literal, OneOf, Sequence
from parsimonious.grammar import Grammar
from parsimonious.nodes import Node, NodeVisitor
from six import reraise
from SpiderEvaluation import get_tables_with_alias, parse_sql

WHITESPACE_REGEX = re.compile(" wsp |wsp | wsp| ws |ws | ws")


def format_grammar_string(grammar_dictionary: Dict[str, List[str]]) -> str:
    """
    Formats a dictionary of production rules into the string format expected
    by the Parsimonious Grammar class.
    """
    return '\n'.join([f"{nonterminal} = {' / '.join(right_hand_side)}"
                      for nonterminal, right_hand_side in grammar_dictionary.items()])


def initialize_valid_actions(grammar: Grammar,
                             keywords_to_uppercase: List[str] = None) -> Dict[str, List[str]]:
    """
    We initialize the valid actions with the global actions. These include the
    valid actions that result from the grammar and also those that result from
    the tables provided. The keys represent the nonterminals in the grammar
    and the values are lists of the valid actions of that nonterminal.
    """
    valid_actions: Dict[str, Set[str]] = defaultdict(set)

    for key in grammar:
        rhs = grammar[key]

        # Sequence represents a series of expressions that match pieces of the text in order.
        # Eg. A -> B C
        if isinstance(rhs, Sequence):
            valid_actions[key].add(
                format_action(key, " ".join(rhs._unicode_members()),  # pylint: disable=protected-access
                              keywords_to_uppercase=keywords_to_uppercase))

        # OneOf represents a series of expressions, one of which matches the text.
        # Eg. A -> B / C
        elif isinstance(rhs, OneOf):
            for option in rhs._unicode_members():  # pylint: disable=protected-access
                valid_actions[key].add(format_action(key, option,
                                                     keywords_to_uppercase=keywords_to_uppercase))

        # A string literal, eg. "A"
        elif isinstance(rhs, Literal):
            if rhs.literal != "":
                valid_actions[key].add(format_action(key, repr(rhs.literal),
                                                     keywords_to_uppercase=keywords_to_uppercase))
            else:
                valid_actions[key] = set()

    valid_action_strings = {key: sorted(value) for key, value in valid_actions.items()}
    return valid_action_strings


def format_action(nonterminal: str,
                  right_hand_side: str,
                  is_string: bool = False,
                  is_number: bool = False,
                  keywords_to_uppercase: List[str] = None) -> str:
    """
    This function formats an action as it appears in models. It
    splits productions based on the special `ws` and `wsp` rules,
    which are used in grammars to denote whitespace, and then
    rejoins these tokens a formatted, comma separated list.
    Importantly, note that it `does not` split on spaces in
    the grammar string, because these might not correspond
    to spaces in the language the grammar recognises.
    Parameters
    ----------
    nonterminal : ``str``, required.
        The nonterminal in the action.
    right_hand_side : ``str``, required.
        The right hand side of the action
        (i.e the thing which is produced).
    is_string : ``bool``, optional (default = False).
        Whether the production produces a string.
        If it does, it is formatted as ``nonterminal -> ['string']``
    is_number : ``bool``, optional, (default = False).
        Whether the production produces a string.
        If it does, it is formatted as ``nonterminal -> ['number']``
    keywords_to_uppercase: ``List[str]``, optional, (default = None)
        Keywords in the grammar to uppercase. In the case of sql,
        this might be SELECT, MAX etc.
    """
    keywords_to_uppercase = keywords_to_uppercase or []
    if right_hand_side.upper() in keywords_to_uppercase:
        right_hand_side = right_hand_side.upper()

    if is_string:
        return f'{nonterminal} -> ["\'{right_hand_side}\'"]'

    elif is_number:
        return f'{nonterminal} -> ["{right_hand_side}"]'

    else:
        right_hand_side = right_hand_side.lstrip("(").rstrip(")")
        child_strings = [token for token in WHITESPACE_REGEX.split(right_hand_side) if token]
        child_strings = [tok.upper() if tok.upper() in keywords_to_uppercase else tok for tok in child_strings]
        return f"{nonterminal} -> [{', '.join(child_strings)}]"


def action_sequence_to_sql(action_sequences: List[str], add_table_names: bool=False) -> str:
    # Convert an action sequence like ['statement -> [query, ";"]', ...] to the
    # SQL string.
    query = []
    for action in action_sequences:
        nonterminal, right_hand_side = action.split(' -> ')
        right_hand_side_tokens = right_hand_side[1:-1].split(', ')
        if nonterminal == 'statement':
            query.extend(right_hand_side_tokens)
        else:
            for query_index, token in list(enumerate(query)):
                if token == nonterminal:
                    if nonterminal == 'column_name' and '@' in right_hand_side_tokens[0] and len(right_hand_side_tokens) == 1:
                        if add_table_names:
                            table_name, column_name = right_hand_side_tokens[0].split('@')
                            if '.' in table_name:
                                table_name = table_name.split('.')[0]
                            right_hand_side_tokens = [table_name + '.' + column_name]
                        else:
                            right_hand_side_tokens = [right_hand_side_tokens[0].split('@')[-1]]
                    query = query[:query_index] + \
                            right_hand_side_tokens + \
                            query[query_index + 1:]
                    break
    return ' '.join([token.strip('"') for token in query])


class SqlVisitor(NodeVisitor):
    """
    ``SqlVisitor`` performs a depth-first traversal of the the AST. It takes the parse tree
    and gives us an action sequence that resulted in that parse. Since the visitor has mutable
    state, we define a new ``SqlVisitor`` for each query. To get the action sequence, we create
    a ``SqlVisitor`` and call parse on it, which returns a list of actions. Ex.
        sql_visitor = SqlVisitor(grammar_string)
        action_sequence = sql_visitor.parse(query)
    Importantly, this ``SqlVisitor`` skips over ``ws`` and ``wsp`` nodes,
    because they do not hold any meaning, and make an action sequence
    much longer than it needs to be.
    Parameters
    ----------
    grammar : ``Grammar``
        A Grammar object that we use to parse the text.
    keywords_to_uppercase: ``List[str]``, optional, (default = None)
        Keywords in the grammar to uppercase. In the case of sql,
        this might be SELECT, MAX etc.
    """

    def __init__(self, grammar: Grammar, keywords_to_uppercase: List[str] = None) -> None:
        self.action_sequence: List[str] = []
        self.grammar: Grammar = grammar
        self.keywords_to_uppercase = keywords_to_uppercase or []

    @overrides
    def generic_visit(self, node: Node, visited_children: List[None]) -> List[str]:
        self.add_action(node)
        if node.expr.name == 'statement':
            return self.action_sequence
        return []

    def add_action(self, node: Node) -> None:
        """
        For each node, we accumulate the rules that generated its children in a list.
        """
        if node.expr.name and node.expr.name not in ['ws', 'wsp']:
            nonterminal = f'{node.expr.name} -> '

            if isinstance(node.expr, Literal):
                right_hand_side = f'["{node.text}"]'

            else:
                child_strings = []
                for child in node.__iter__():
                    if child.expr.name in ['ws', 'wsp']:
                        continue
                    if child.expr.name != '':
                        child_strings.append(child.expr.name)
                    else:
                        child_right_side_string = child.expr._as_rhs().lstrip("(").rstrip(
                            ")")  # pylint: disable=protected-access
                        child_right_side_list = [tok for tok in
                                                 WHITESPACE_REGEX.split(child_right_side_string) if tok]
                        child_right_side_list = [tok.upper() if tok.upper() in
                                                                self.keywords_to_uppercase else tok
                                                 for tok in child_right_side_list]
                        child_strings.extend(child_right_side_list)
                right_hand_side = "[" + ", ".join(child_strings) + "]"
            rule = nonterminal + right_hand_side
            self.action_sequence = [rule] + self.action_sequence

    @overrides
    def visit(self, node):
        """
        See the ``NodeVisitor`` visit method. This just changes the order in which
        we visit nonterminals from right to left to left to right.
        """
        method = getattr(self, 'visit_' + node.expr_name, self.generic_visit)

        # Call that method, and show where in the tree it failed if it blows
        # up.
        try:
            # Changing this to reverse here!
            return method(node, [self.visit(child) for child in reversed(list(node))])
        except (VisitationError, UndefinedLabel):
            # Don't catch and re-wrap already-wrapped exceptions.
            raise
        except self.unwrapped_exceptions:
            raise
        except Exception:  # pylint: disable=broad-except
            # Catch any exception, and tack on a parse tree so it's easier to
            # see where it went wrong.
            exc_class, exc, traceback = exc_info()
            reraise(VisitationError, VisitationError(exc, exc_class, node), traceback)

class TableColumn:
    def __init__(self,
                 name: str,
                 text: str,
                 column_type: str,
                 is_primary_key: bool,
                 foreign_key: Optional[str]):
        self.name = name
        self.text = text
        self.column_type = column_type
        self.is_primary_key = is_primary_key
        self.foreign_key = foreign_key

class Table:
    def __init__(self,
                 name: str,
                 text: str,
                 columns: List[TableColumn]):
        self.name = name
        self.text = text
        self.columns = columns

def read_dataset_schema(schema_path: str) -> Dict[str, List[Table]]:
    schemas: Dict[str, Dict[str, Table]] = defaultdict(dict)
    dbs_json_blob = json.load(open(schema_path, "r"))
    for db in dbs_json_blob:
        db_id = db['db_id']

        column_id_to_table = {}
        column_id_to_column = {}

        for i, (column, text, column_type) in enumerate(
                zip(db['column_names_original'], db['column_names'], db['column_types'])):
            table_id, column_name = column
            _, column_text = text

            table_name = db['table_names_original'][table_id]

            if table_name not in schemas[db_id]:
                table_text = db['table_names'][table_id]
                schemas[db_id][table_name] = Table(table_name, table_text, [])

            if column_name == "*":
                continue

            is_primary_key = i in db['primary_keys']
            table_column = TableColumn(column_name.lower(), column_text, column_type, is_primary_key, None)
            schemas[db_id][table_name].columns.append(table_column)
            column_id_to_table[i] = table_name
            column_id_to_column[i] = table_column

        for (c1, c2) in db['foreign_keys']:
            foreign_key = column_id_to_table[c2] + ':' + column_id_to_column[c2].name
            column_id_to_column[c1].foreign_key = foreign_key

    return {**schemas}

def ent_key_to_name(key):
    parts = key.split(':')
    if parts[0] == 'table':
        return parts[1]
    elif parts[0] == 'column':
        _, _, table_name, column_name = parts
        return f'{table_name}@{column_name}'
    else:
        return parts[1]

def fix_number_value(ex: JsonDict):
    """
    There is something weird in the dataset files - the `query_toks_no_value` field anonymizes all values,
    which is good since the evaluator doesn't check for the values. But it also anonymizes numbers that
    should not be anonymized: e.g. LIMIT 3 becomes LIMIT 'value', while the evaluator fails if it is not a number.
    """

    def split_and_keep(s, sep):
        if not s: return ['']  # consistent with string.split()

        # Find replacement character that is not used in string
        # i.e. just use the highest available character plus one
        # Note: This fails if ord(max(s)) = 0x10FFFF (ValueError)
        p = chr(ord(max(s)) + 1)

        return s.replace(sep, p + sep + p).split(p)

    # input is tokenized in different ways... so first try to make splits equal
    query_toks = ex['query_toks']
    ex['query_toks'] = []
    for q in query_toks:
        ex['query_toks'] += split_and_keep(q, '.')

    i_val, i_no_val = 0, 0
    while i_val < len(ex['query_toks']) and i_no_val < len(ex['query_toks_no_value']):
        if ex['query_toks_no_value'][i_no_val] != 'value':
            i_val += 1
            i_no_val += 1
            continue

        i_val_end = i_val
        while i_val + 1 < len(ex['query_toks']) and \
                i_no_val + 1 < len(ex['query_toks_no_value']) and \
                ex['query_toks'][i_val_end + 1].lower() != ex['query_toks_no_value'][i_no_val + 1].lower():
            i_val_end += 1

        if i_val == i_val_end and ex['query_toks'][i_val] in ["1", "2", "3"] and ex['query_toks'][
            i_val - 1].lower() == "limit":
            ex['query_toks_no_value'][i_no_val] = ex['query_toks'][i_val]
        i_val = i_val_end

        i_val += 1
        i_no_val += 1

    return ex


_schemas_cache = None


def disambiguate_items(db_id: str, query_toks: List[str], schema) -> List[str]:
    """
    we want the query tokens to be non-ambiguous - so we can change each column name to explicitly
    tell which table it belongs to
    parsed sql to sql clause is based on supermodel.gensql from syntaxsql
    """

    class Schema_dicts:
        """
        Simple schema which maps table&column to a unique identifier
        """

        def __init__(self, schema, table):
            self._schema = schema
            self._table = table
            self._idMap = self._map(self._schema, self._table)

        @property
        def schema(self):
            return self._schema

        @property
        def idMap(self):
            return self._idMap

        def _map(self, schema, table):
            column_names_original = table['column_names_original']
            table_names_original = table['table_names_original']
            # print 'column_names_original: ', column_names_original
            # print 'table_names_original: ', table_names_original
            for i, (tab_id, col) in enumerate(column_names_original):
                if tab_id == -1:
                    idMap = {'*': i}
                else:
                    key = table_names_original[tab_id].lower()
                    val = col.lower()
                    idMap[key + "." + val] = i

            for i, tab in enumerate(table_names_original):
                key = tab.lower()
                idMap[key] = i

            return idMap

    def schema_to_dicts(schema):
        schema_dict = {}  # {'table': [col.lower, ..., ]} * -> __all__
        column_names_original = schema.column_names_original
        table_names_original = schema.table_names_original
        tables = {'column_names_original': column_names_original,
                  'table_names_original': table_names_original}
        for i, tabn in enumerate(table_names_original):
            table = str(tabn.lower())
            cols = [str(col.lower()) for td, col in column_names_original if td == i]
            schema_dict[table] = cols

        return schema_dict, tables

    schema_dict, tables_dict = schema_to_dicts(schema)
    schema_dicts = Schema_dicts(schema_dict, tables_dict)

    fixed_toks = []
    i = 0
    while i < len(query_toks):
        tok = query_toks[i]
        if tok == 'value' or tok == "'value'":
            # TODO: value should alawys be between '/" (remove first if clause)
            new_tok = f'"{tok}"'
        elif tok in ['!', '<', '>'] and query_toks[i + 1] == '=':
            new_tok = tok + '='
            i += 1
        elif i + 1 < len(query_toks) and query_toks[i + 1] == '.':
            new_tok = ''.join(query_toks[i:i + 3])
            i += 2
        else:
            new_tok = tok
        fixed_toks.append(new_tok)
        i += 1

    toks = fixed_toks

    tables_with_alias = get_tables_with_alias(schema_dicts.schema, toks)
    _, sql, mapped_entities = parse_sql(toks, 0, tables_with_alias, schema_dicts, mapped_entities_fn=lambda: [])

    for i, new_name in mapped_entities:
        curr_tok = toks[i]
        toks[i] = new_name

    toks = [tok for tok in toks if tok not in ['as', 't1', 't2', 't3', 't4']]

    toks = [f'\'value\'' if tok == '"value"' else tok for tok in toks]

    return toks