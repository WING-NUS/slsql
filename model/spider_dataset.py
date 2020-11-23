from __future__ import absolute_import, division, print_function

import json
import logging
import os
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)

feature_list = ['NUMBER', 'DATE', 'QUOTE'] + ['none', '*', '#']

T_KWS = ['FROM']
C_KWS = ['SELECT', 'WHERE', 'HAVING', 'GROUP_BY', 'ORDER_BY']
O_KWS = ['asc', 'desc']
V_KWS = ['LIMIT']
N_KWS = ['(', ')']
R_KWS = ['and', 'or']
CONJ = [',', 'JOIN']
IUE = ['INTERSECT', 'EXCEPT', 'UNION']

OPS = ['between', '=', '>', '<', '>=', '<=', '!=', 'in', 'not', 'like']

TASK_KWS = T_KWS + C_KWS
ALL_KWS = ['EOS'] + T_KWS + C_KWS + O_KWS + V_KWS + N_KWS + R_KWS + CONJ + IUE + ['SOS'] + ['value'] + OPS
ALL_KWS_MAP = {k: v for v, k in enumerate(ALL_KWS)}
AGG_OPS = ['max', 'min', 'count', 'sum', 'avg']


def feature_name(name):
    return '[{}]'.format(name)


def make_features(tokenizer):
    for id, tok in enumerate(feature_list):
        token = '[unused{}]'.format(id)
        new_token = feature_name(tok)
        assert token in tokenizer.vocab
        tokenizer.vocab[new_token] = tokenizer.vocab[token]
        tokenizer.basic_tokenizer.never_split += (new_token,)
        del tokenizer.vocab[token]


class InputExample(object):
    def __init__(self, guid, input_seq, labels, extra):
        self.guid = guid
        self.input_seq = input_seq
        self.labels = labels
        self.extra = extra


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, labels):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.labels = labels


def alt_tbl_name(tbl_name):
    tbl_name = tbl_name.split()
    if len(tbl_name) > 1 and tbl_name[0] == 'reference':
        tbl_name = tbl_name[1:]
    if len(tbl_name) > 1 and tbl_name[-1] == 'data':
        tbl_name = tbl_name[:-1]
    if len(tbl_name) > 1 and tbl_name[-1] == 'list':
        tbl_name = tbl_name[:-1]
    return ' '.join(tbl_name)


def fix_tokens(sample):
    def fix_tok(tok):
        def is_number(s):
            try:
                float(s)
                return True
            except ValueError:
                return False

        tok = tok.lower()
        changed = True
        if tok == '-lrb-':
            tok = '('
        elif tok == '-rrb-':
            tok = ')'
        elif tok in ["''", "'", '"', "‘", "’"]:
            tok = '"'
        elif tok.isdigit() or is_number(tok):
            if len(tok) == 4 and tok.isdigit() and 1800 <= int(tok) <= 2100:
                tok = feature_name('DATE')
            else:
                tok = feature_name('NUMBER')
        else:
            changed = False
        return tok, changed

    toks = sample['toks']
    lemma = sample['lemma']
    question = sample['question']
    i = 0
    while i < len(toks):
        if toks[i].lower() in ['the', 'id', 'states', 'country', 'city']:
            toks[i] = toks[i].lower()
        tok = toks[i]
        if tok in ["''", "'", '"', "``", "‘", "`"]:
            ok = False
            end = None
            for j, tok2 in enumerate(toks[i + 1:], i + 1):
                if (tok2 == tok or (tok2 == "''" and tok == '``') or (tok2 == "’" and tok == '‘')) and not ' ' + toks[
                    i + 1] in question:
                    ok = True
                    end = j
                    break
            if ok:
                toks[i], _ = fix_tok(toks[i])
                lemma[i] = toks[i]
                toks[end], _ = fix_tok(toks[end])
                lemma[end] = toks[end]

                for j in range(i + 1, end):
                    toks[j] = feature_name('QUOTE')
                    lemma[j] = toks[j]
                i = end
        else:
            if i < len(toks):
                toks[i], changed = fix_tok(toks[i])
                if changed:
                    lemma[i] = toks[i]
        i += 1
    return toks


class SpiderDataset:
    def get_train_examples(self, data_dir):
        tables = self._read_json(os.path.join(data_dir, 'processed_tables.json'))
        tables = {table['db_id']: table for table in tables}
        data = self._read_json(os.path.join(data_dir, 'slsql_train.json'))
        return self._create_examples(tables, data)

    def get_dev_examples(self, data_dir):
        tables = self._read_json(os.path.join(data_dir, 'processed_tables.json'))
        tables = {table['db_id']: table for table in tables}
        data = self._read_json(os.path.join(data_dir, 'slsql_dev.json'))
        return self._create_examples(tables, data, mode='dev')

    def _read_json(self, file):
        with open(file) as f:
            return json.load(f)

    def tbl_name(self, id, db):
        return db['table_names_lemma'][id]

    def refine(self, tbl_name, col_name, col_id, db):
        if col_id in [x[0] for x in db['foreign_keys']] or col_id == 0:
            return col_name
        tbl_name = tbl_name.split()
        col_name = col_name.split()

        for st in range(len(col_name) - 1, -1, -1):
            if col_name[:st] == tbl_name[-st:]:
                return ' '.join(col_name[st:])
        return ' '.join(col_name)

    def col_full_name(self, id, db, ub=None):
        def convert(tok):
            if tok == 'num':
                tok = 'number'
            return tok

        col_name_lemma = db['column_names_lemma'][id][1]
        col_name_lemma = col_name_lemma.split()
        col_name_lemma = [convert(x) for x in col_name_lemma]
        if ub is not None and len(col_name_lemma) > ub:
            col_name_lemma = col_name_lemma[-ub:]
        col_name_lemma = ' '.join(col_name_lemma)
        table_id = db['column_names_lemma'][id][0]
        if table_id == -1:
            return col_name_lemma
        else:
            tbl_name = db['table_names_lemma'][table_id]
            col_name = self.refine(tbl_name, col_name_lemma, id, db)
            tbl_name = alt_tbl_name(tbl_name)
            return '{} [#] {}'.format(tbl_name, col_name)

    def _foreach_example(self, s_id, sample, db, mode='train'):
        fix_tokens(sample)

        col_num = len(db['column_names_lemma'])
        tbl_num = len(db['table_names_lemma'])
        parsed_question = ' '.join(sample['lemma'])
        input_seq = []
        input_seq.append(parsed_question)
        input_seq.append((0, '[*]'))
        for ti, tbl_name in enumerate(db['table_names_lemma']):
            input_seq.append((ti + col_num, tbl_name))
            for ci, col in enumerate(db['column_names_lemma']):
                tbl_id, col_name = col
                if tbl_id != ti:
                    continue
                input_seq.append((ci, self.col_full_name(ci, db)))
        input_seq.append((col_num + tbl_num, '[none]'))

        ct_linking = []
        val_linking = []
        assert len(sample['ant']) == len(sample['lemma'])
        for sl in sample['ant']:
            if sl is None:
                ct_linking.append(col_num + tbl_num)
                val_linking.append(col_num + tbl_num)
            elif sl['type'] == 'col':
                ct_linking.append(sl['id'])
                assert sl['id'] < col_num
                val_linking.append(-100)
            elif sl['type'] == 'tbl':
                ct_linking.append(sl['id'] + col_num)
                assert sl['id'] < tbl_num
                val_linking.append(-100)
            else:
                ct_linking.append(col_num + tbl_num)
                val_linking.append(sl['id'])
                assert sl['id'] < col_num

        linking_labels = {
            "ct_linking": ct_linking,
            "val_linking": val_linking
        }
        extra = {
            "col_num": col_num,
            "tbl_num": tbl_num,
            "db": db,
            "lemma": sample['lemma']
        }
        exp = InputExample(s_id, input_seq, linking_labels, extra)
        if mode == 'train':
            seq, agg = self._convert_seq(sample['seq'], db)
            exp.labels['seq'] = seq
            exp.labels['agg'] = agg
        return exp

    def _create_examples(self, dbs, data, mode='train'):
        examples = []
        for s_id, sample in enumerate(data, 1):
            # these dbs contained some issues
            if sample['db_id'] in ['formula_1', 'baseball_1', 'cre_Drama_Workshop_Groups', 'soccer_1']:
                continue
            exp = self._foreach_example(s_id, sample, dbs[sample['db_id']], mode)
            if exp is None:
                continue
            examples.append(exp)
        return examples

    def _foreach_feature(self, example, max_seq_length, tokenizer, mode='train'):
        col_num = example.extra['col_num']
        tbl_num = example.extra['tbl_num']
        db = example.extra['db']

        tokens = ["[CLS]"]
        orig_to_tok_map = {}
        que_toks = example.input_seq[0].split()
        for i, (token) in enumerate((que_toks)):
            orig_to_tok_map[i] = len(tokens)
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tokens.append(sub_token)
        tokens.append("[SEP]")

        segment_ids = [0] * len(tokens)

        item_pos_map = {}
        for (seg_id, seg) in example.input_seq[1:]:
            item_pos_map[seg_id] = []
            seg_toks = seg.split()
            for st in seg_toks:
                item_pos_map[seg_id].append(len(tokens))
                st_tokens = tokenizer.tokenize(st)
                tokens += st_tokens
            tokens.append("[SEP]")

        segment_ids += [1] * (len(tokens) - len(segment_ids))
        assert len(segment_ids) == len(tokens)
        if len(tokens) > 512:
            return None
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        assert len(input_ids) == len(input_mask) == len(segment_ids) == max_seq_length

        item_pos = []
        for i in range(col_num + tbl_num + 1):
            pos = item_pos_map[i]
            item_pos.append(pos)

        pks = [x[1] for x in db['foreign_keys']]
        fks = [x[0] for x in db['foreign_keys']]
        key_info = []
        for i in range(col_num):
            pk = int(i in db['primary_keys'] or i in pks)
            fk = int(i in fks)
            key_info.append((pk, fk))

        label = {
            "ct_linking": example.labels['ct_linking'],
            "val_linking": example.labels['val_linking'],
            "col_num": col_num,
            "tbl_num": tbl_num,
            "item_pos": item_pos,
            "que_tok_pos": list(orig_to_tok_map.values()),
            "key_info": key_info,
            "belongs": [x[0] for x in example.extra['db']['column_names_lemma']],
            "key_pairs": db['foreign_keys'] if len(db['foreign_keys']) else [[-100, -100]]
        }
        example.extra["orig_to_tok_map"] = orig_to_tok_map
        example.extra["item_to_tok_map"] = item_pos_map
        fea = InputFeatures(input_ids, input_mask, segment_ids, label)
        if mode == 'train':
            fea.labels['seq'] = example.labels['seq']
            fea.labels['agg'] = example.labels['agg']
        return fea

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer, mode='train'):
        ret = []
        for (idx, example) in enumerate(examples):
            if idx % 100 == 0:
                logger.info("Writing example %d of %d" % (idx, len(examples)))
            fea = self._foreach_feature(example, max_seq_length, tokenizer, mode)
            ret.append(fea)
        return ret

    def read_seq(self, seq, col_num):

        def extract_col_agg(s, agg_pos):
            r = []
            for i, col in enumerate(s):
                if agg_pos is not None and len(col[agg_pos]) > 1 and len(set(list(col[agg_pos]))) == 1:
                    col[agg_pos] = ['none']
                if agg_pos is None:
                    r.append((col[0]))
                elif col[agg_pos] == ['none']:
                    r.append((col[0], []))
                else:
                    # self join has bug
                    aggs = [AGG_OPS.index(x) for x in col[agg_pos]]
                    r.append((col[0], aggs))
                if i != len(s) - 1:
                    r.append(',')
            return r

        def pt(now, ret):
            for each in now:
                if each[0] == 'SELECT':
                    ret.append('SELECT')
                    ret += extract_col_agg(each[1], 2)
                elif each[0] == 'FROM':
                    ret.append('FROM')
                    for i, tbl in enumerate(each[1]):
                        ret.append(tbl[0] + col_num)
                        if i != len(each[1]) - 1:
                            ret.append('JOIN')
                elif each[0] in ['WHERE', 'HAVING']:
                    ret.append(each[0])
                    for thing in each[1]:
                        if thing in ['and', 'or']:
                            ret.append(thing)
                        else:
                            # id
                            if each[0] == 'WHERE':
                                ret.append(thing[0][0])
                            else:
                                ret.append((thing[0][0], AGG_OPS.index(thing[0][2])))
                            ret += thing[1].split('_')
                            if thing[2] == 'value':
                                ret.append(thing[2])
                            else:
                                ret.append('(')
                                pt(thing[2], ret)
                                ret.append(')')
                elif each[0] == 'GROUP_BY':
                    ret.append('GROUP_BY')
                    ret += extract_col_agg(each[1], None)
                elif each[0] == 'ORDER_BY':
                    ret.append('ORDER_BY')
                    for i, col in enumerate(each[1]):
                        agg = None if col[0][2] == 'none' else AGG_OPS.index(col[0][2])
                        ret.append((col[0][0], agg))
                        ret.append(col[1])
                        if i != len(each[1]) - 1:
                            ret.append(',')
                elif each[0] == 'LIMIT':
                    ret.append('LIMIT')
                    ret.append('value')
                else:
                    # IUE
                    ret.append(each)

        ret = []
        pt(seq, ret)
        return ret

    def _convert_seq(self, seq, db):
        col_num = len(db['column_names'])
        seq = self.read_seq(seq, col_num)
        tok_ids = []
        aggs = []
        for i, tok in enumerate(seq + ['EOS']):
            if tok in ALL_KWS:
                tok_id = ALL_KWS_MAP[tok]
                agg = []
            else:
                if type(tok) is tuple:
                    tok_id = tok[0]
                    if type(tok[1]) == list:
                        agg = tok[1]
                    elif type(tok[1]) == int:
                        agg = [tok[1]]
                    elif tok[1] is None:
                        agg = []
                    else:
                        assert False
                else:
                    tok_id = tok
                    agg = []
                tok_id += len(ALL_KWS)

            tok_ids.append(tok_id)
            agg = [1 if i in agg else 0 for i in range(len(AGG_OPS))]
            aggs.append(agg)
        assert len(tok_ids) == len(aggs)
        return tok_ids, aggs

    def print_result(self, data_dir, pred_links, pred_sqls):
        tables = self._read_json(os.path.join(data_dir, 'processed_tables.json'))
        tables = {table['db_id']: table for table in tables}
        data = self._read_json(os.path.join(data_dir, 'slsql_dev.json'))
        ret = []
        for idx, (sample, pred_link, pred_sql) in enumerate(
                zip(data, pred_links, pred_sqls)):
            assert type(tables) == dict
            db = tables[sample['db_id']]
            col_num = len(db['column_names_lemma'])
            ant = sample['ant']
            preds = []
            for i, p in enumerate(pred_link):
                if p is None:
                    preds.append(None)
                    continue
                name = ''
                if p[1] == 'col':
                    name = self.col_full_name(p[0], db)
                elif p[1] == 'tbl':
                    name = self.tbl_name(p[0] - col_num, db)
                elif p[1] == 'val':
                    name = self.col_full_name(p[0], db)
                cur = (p[0], name, p[1])
                preds.append(cur)
            assemble = []
            for tok, fl, mp in zip(sample['toks'], ant, preds):
                assemble.append((tok, fl, mp))
            ret.append({
                "id": sample['id'],
                "question": sample['question'],
                "query": sample['query'],
                "links": assemble,
                "sql": ' '.join(pred_sql), }
            )
        return ret


def trans(task, tok, col_num, tbl_num, has_left=False):
    col_ran = np.r_[len(ALL_KWS):len(ALL_KWS) + col_num]
    tbl_ran = np.r_[len(ALL_KWS) + col_num:len(ALL_KWS) + col_num + tbl_num]
    op_ran = np.r_[len(ALL_KWS) - len(OPS):len(ALL_KWS)]
    iue_ran = np.r_[ALL_KWS_MAP['INTERSECT'], ALL_KWS_MAP['UNION'], ALL_KWS_MAP['EXCEPT']]
    end_ran = np.r_[ALL_KWS_MAP['EOS'], iue_ran]
    if tok in C_KWS or tok == ',':
        # must be column
        return col_ran, 'col'
    elif tok in T_KWS or tok == 'JOIN':
        # must be table
        return tbl_ran, 'tbl'
    elif tok in O_KWS:
        ret = np.r_[ALL_KWS_MAP[','], ALL_KWS_MAP['LIMIT'], end_ran]
        return np.r_[ret, ALL_KWS_MAP[')']] if has_left else ret, 'kw'
    elif tok in V_KWS:
        return np.r_[ALL_KWS_MAP['value']], 'kw'
    elif tok in N_KWS:
        if tok == '(':
            return np.r_[ALL_KWS_MAP['SELECT']], 'kw'
        else:
            ret = np.r_[ALL_KWS_MAP['GROUP_BY'], ALL_KWS_MAP['ORDER_BY'], end_ran]
            if task in ['WHERE', 'HAVING']:
                ret = np.r_[ret, ALL_KWS_MAP['and'], ALL_KWS_MAP['or']]
            return np.r_[ret, ALL_KWS_MAP[')']] if has_left else ret, 'kw'
    elif tok in R_KWS:
        # column
        return col_ran, 'col'
    elif tok in IUE:
        return np.r_[ALL_KWS_MAP['SELECT']], 'kw'
    elif tok == 'col':
        # FROM , and op
        # only order_by group_by can end
        if task == 'SELECT':
            return np.r_[ALL_KWS_MAP['FROM'], ALL_KWS_MAP[',']], 'kw'
        elif task == 'GROUP_BY':
            ret = np.r_[ALL_KWS_MAP['HAVING'], ALL_KWS_MAP['ORDER_BY'], ALL_KWS_MAP[','], end_ran]
            return np.r_[ret, ALL_KWS_MAP[')']] if has_left else ret, 'kw'
        elif task == 'ORDER_BY':
            return np.r_[ALL_KWS_MAP['asc'], ALL_KWS_MAP['desc']], 'kw'
        elif task in ['WHERE', 'HAVING']:
            return op_ran, 'kw'
        assert False
    elif tok in OPS:
        if tok == 'not':
            return np.r_[ALL_KWS_MAP['like'], ALL_KWS_MAP['in']], 'kw'
        elif tok in OPS[1:8]:
            return np.r_[ALL_KWS_MAP['('], ALL_KWS_MAP['value']], 'kw'
        elif tok in ['between', 'like']:
            return np.r_[ALL_KWS_MAP['value']], 'kw'
        assert False
    elif tok == 'tbl':
        ret = np.r_[ALL_KWS_MAP['JOIN'], ALL_KWS_MAP['WHERE'], ALL_KWS_MAP['GROUP_BY'], ALL_KWS_MAP[
            'ORDER_BY'], end_ran]
        return np.r_[ret, ALL_KWS_MAP[')']] if has_left else ret, 'kw'
    elif tok == 'value':
        if task == 'ORDER_BY':
            return np.r_[end_ran, ALL_KWS_MAP[')']] if has_left else end_ran, 'kw'
        if task == 'HAVING':
            ret = np.r_[ALL_KWS_MAP['and'], ALL_KWS_MAP['or'], ALL_KWS_MAP[
                'ORDER_BY'], end_ran]
        else:
            ret = np.r_[ALL_KWS_MAP['and'], ALL_KWS_MAP['or'], ALL_KWS_MAP['GROUP_BY'], ALL_KWS_MAP[
                'ORDER_BY'], end_ran]
        return np.r_[ret, ALL_KWS_MAP[')']] if has_left else ret, 'kw'
    else:
        assert tok == 'EOS'
        return np.r_[ALL_KWS_MAP['EOS']], 'kw'


def edit_sql(step_id, seq, agg, db, col_num, tbl_num):
    def tbl_con(a, b, db, rel_check=False):
        for fk in db['foreign_keys']:
            ca, cb = fk[0], fk[1]
            ta, tb = db['column_names'][ca][0], db['column_names'][cb][0]
            if not rel_check and ((a, b) == (ta, tb) or (a, b) == (tb, ta)):
                return True
            if rel_check and (a, b) == (ta, tb):
                return True
        return False

    def remove_all(tbls, tbl_funs, db, sub_have_tbls):

        def is_rel(x, y, z):
            r, a, b = tbls[x], tbls[y], tbls[z]
            return tbl_con(r, a, db, True) and tbl_con(r, b, db, True)

        def do_rm(x, y, z):
            r, a, b = tbls[x], tbls[y], tbls[z]
            if len(tbl_funs[a]) == 0:
                return [r, b]
            if len(tbl_funs[b]) == 0:
                return [r, a]
            return [r, a, b]

        if len(tbls) == 3:
            if is_rel(0, 1, 2):
                ret = do_rm(0, 1, 2)
                if len(ret) == 2:
                    return remove_all(ret, tbl_funs, db, sub_have_tbls)
                return ret
            elif is_rel(1, 0, 2):
                ret = do_rm(1, 0, 2)
                if len(ret) == 2:
                    return remove_all(ret, tbl_funs, db, sub_have_tbls)
                return ret
            elif is_rel(2, 0, 1):
                ret = do_rm(2, 0, 1)
                if len(ret) == 2:
                    return remove_all(ret, tbl_funs, db, sub_have_tbls)
                return ret
        elif len(tbls) == 2:
            if len(tbl_funs[tbls[0]]) and len(tbl_funs[tbls[1]]):
                return tbls
            if len(tbl_funs[tbls[0]]):
                if tbls[1] in sub_have_tbls:
                    return [tbls[0]]
            if len(tbl_funs[tbls[1]]):
                if tbls[0] in sub_have_tbls:
                    return [tbls[1]]
            if tbl_con(tbls[0], tbls[1], db):
                return tbls
            return [tbls[0]] if len(tbl_funs[tbls[0]]) else [tbls[1]]

        return tbls

    def fix_more(x, y, db, tbl_num):
        if tbl_con(x, y, db):
            return []
        for t in range(tbl_num):
            if t in [x, y]:
                continue
            if tbl_con(x, t, db) and tbl_con(t, y, db):
                return [t]
        for t in range(tbl_num):
            if t in [x, y]:
                continue
            if tbl_con(x, t, db) and tbl_con(t, y, db):
                return [t]

        for t1 in range(tbl_num):
            if t1 in [x, y]:
                continue
            for t2 in range(tbl_num):
                if t2 in [x, y]:
                    continue
                if tbl_con(x, t1, db) and tbl_con(y, t2, db) and tbl_con(t1, t2, db):
                    return [t1, t2]
        return []

    def gen_from_clause(tbls, col_num, db):
        from_clause = [tbls[0] + len(ALL_KWS) + col_num]
        for i, tbl in enumerate(tbls[1:], 1):
            from_clause.append(ALL_KWS_MAP['JOIN'])
            from_clause.append(tbl + len(ALL_KWS) + col_num)
            for fk in db['foreign_keys']:
                ca, cb = fk[0], fk[1]
                ta, tb = db['column_names_original'][ca][0], db['column_names_original'][cb][0]
                if (ta, tb) == (tbls[i], tbls[i - 1]) or (tb, ta) == (tbls[i], tbls[i - 1]):
                    nta, ntb = db['table_names_original'][ta], db['table_names_original'][tb]
                    nca, ncb = db['column_names_original'][ca][1], db['column_names_original'][cb][1]
                    cond1 = '{}.{}'.format(nta, nca)
                    cond2 = '{}.{}'.format(ntb, ncb)
                    from_clause.append(' on {} = {} '.format(cond1, cond2))
                    break
        return from_clause

    offset = step_id
    ret = []
    have_tbls = set()
    should_tbls = set()
    tbl_funs = defaultdict(lambda: defaultdict(set))
    r_seq = []
    r_agg = []
    from_pos = []
    tasks = [('SELECT', set())]
    sub_have_tbls = []
    while step_id < len(seq):
        tok_id = seq[step_id]
        if tok_id < len(ALL_KWS):
            pass
        elif tok_id < len(ALL_KWS) + col_num:
            tbl_id, _ = tuple(db['column_names_original'][tok_id - len(ALL_KWS)])
            if tok_id != len(ALL_KWS):
                tbl_funs[tbl_id][tasks[-1][0]].add(tok_id - len(ALL_KWS))
                should_tbls.add(tbl_id)
            ret.append(tok_id)
        else:
            have_tbls.add(tok_id - len(ALL_KWS) - col_num)
        r_seq.append(tok_id)
        r_agg.append(agg[step_id])
        if tok_id == ALL_KWS_MAP['SELECT']:
            step_id, s_seq, s_agg, s_tbls = edit_sql(step_id + 1, seq, agg, db, col_num, tbl_num)
            r_seq += s_seq
            r_agg += s_agg
            sub_have_tbls += s_tbls
        elif tok_id != [ALL_KWS_MAP['EOS']]:
            if tok_id < len(ALL_KWS) and ALL_KWS[tok_id] in TASK_KWS:
                tasks.append((ALL_KWS[tok_id], set()))
            if tok_id >= len(ALL_KWS) + col_num:
                from_pos.append(step_id)
            step_id += 1
        if tok_id in [ALL_KWS_MAP['EOS'], ALL_KWS_MAP[')']]:
            break

    if from_pos == []:
        assert len(r_seq) == len(r_agg)
        return step_id, r_seq, r_agg, have_tbls

    have_tbls.update(should_tbls)
    if len(have_tbls) == 2:
        have_list = list(have_tbls)
        more = fix_more(have_list[0], have_list[1], db, tbl_num)
        have_tbls.update(more)
    have_tbls = list(have_tbls)
    have_tbls = remove_all(have_tbls, tbl_funs, db, sub_have_tbls)

    from_clause = gen_from_clause(have_tbls, col_num, db)
    start = min(from_pos) - offset
    end = max(from_pos) + 1 - offset
    r_seq = r_seq[:start] + from_clause + r_seq[end:]
    r_agg = r_agg[:start] + [None] * len(from_clause) + r_agg[end:]
    assert len(r_seq) == len(r_agg)
    return step_id, r_seq, r_agg, have_tbls


def write_sql(seqs, aggs, col_num, db, eg):
    ret = ['SELECT']
    for seq, agg in zip(seqs, aggs):
        if seq == ALL_KWS_MAP['EOS']:
            break
        if type(seq) == str:
            ret.append(seq)
            continue
        if seq < len(ALL_KWS):
            kw = ALL_KWS[seq].replace('_', ' ')
            if kw == 'value':
                kw = '1'
                if ret[-1] == 'between':
                    kw = '1 and 2'
            elif kw == 'asc' and ('reverse' in eg.extra['lemma'] or 'reversed' in eg.extra['lemma']):
                kw = 'desc'
            ret.append(kw)
        elif seq < len(ALL_KWS) + col_num:
            tbl_id, col = tuple(db['column_names_original'][seq - len(ALL_KWS)])
            if col != '*':
                tbl = db['table_names_original'][tbl_id]
                col = tbl + '.' + col

            if agg is not None:
                ca = [x for x in range(5) if agg[x] >= 0.5]
                ca = [AGG_OPS[x] for x in ca]
                col = ' , '.join(['{}( {} )'.format(a, col) for a in ca]) if len(ca) else col
            ret.append(col)
        else:
            tbl = db['table_names_original'][seq - len(ALL_KWS) - col_num]
            ret.append(tbl)

    return ret
