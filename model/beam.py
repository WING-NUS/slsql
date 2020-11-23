from collections import defaultdict

from spider_dataset import ALL_KWS, TASK_KWS, ALL_KWS_MAP, OPS

INF = 100000000
NINF = -INF


class BeamSearchNode(object):
    def __init__(self, col_num, tbl_num, parent, next_h, tok_id, bracket, prob, length, tok_range, tok_type, tasks,
                 tok_score, agg_mask, dc, dt, ct_hit=None, val_hit=None, agg_input=None, key_pair=None, key_info=None):
        self.h = next_h
        self.parent = parent
        self.tok_id = tok_id
        self.bracket = bracket
        self.prob = prob
        self.penalty = 0
        self.penalty_sum = 0
        self.prob_sum = prob
        if parent is not None:
            self.prob_sum += parent.prob_sum
        self.len = length
        self.tok_range = tok_range
        self.tok_type = tok_type
        self.tasks = tasks
        self.tok_score = tok_score
        self.agg_mask = agg_mask
        self.fob_cols = dc
        self.fob_tbls = dt
        self.col_num = col_num
        self.tbl_num = tbl_num
        self.ct_hit = ct_hit
        self.val_hit = val_hit
        self.agg_input = agg_input
        self.key_pair = key_pair
        self.key_info = key_info
        self.clause_cols = defaultdict(set)
        if self.tok_id - len(ALL_KWS) in range(col_num):
            self.clause_cols[self.tasks[-1][0]].add(self.tok_id - len(ALL_KWS))

    def recover_scores(self):
        tok_scores, agg_inputs = [self.tok_score], [self.agg_input]
        agg_masks = [self.agg_mask]
        next = self.parent
        while next is not None and next.len > 0:
            tok_scores.append(next.tok_score)
            agg_inputs.append(next.agg_input)
            agg_masks.append(next.agg_mask)
            next = next.parent
        agg_masks = [x for x in reversed(agg_masks) if x is not None]
        return list(reversed(tok_scores)), list(reversed(agg_inputs)), agg_masks

    def recover_ret(self):
        seq_rets, agg_inputs = [self.tok_id], [self.agg_input]
        next = self.parent
        while next is not None and next.len > 0:
            seq_rets.append(next.tok_id)
            agg_inputs.append(next.agg_input)
            for task in TASK_KWS:
                self.clause_cols[task] |= next.clause_cols[task]
            next = next.parent
        return list(reversed(seq_rets)), list(reversed(agg_inputs))

    def eval(self, oracle=False, reward=False):
        self.penalty = self.calc_penalty(oracle)
        self.penalty_sum = self.penalty + self.parent.penalty_sum if self.parent is not None else 0
        score = self.prob_sum / (float(self.len - 1 + 1e-6) ** 0.95) - self.penalty_sum
        if oracle and reward:
            score += 5 * self.calc_reward()
        return score

    def seq_ret(self, offset):
        now = self
        for _ in range(offset):
            now = now.parent
        return now.tok_id

    def calc_reward(self):
        score = 0
        seq_ret, agg_ret = self.recover_ret()
        cols = [x - len(ALL_KWS) for x in seq_ret if 0 <= x - len(ALL_KWS) < self.col_num]
        tar_cols = [x for x in self.ct_hit if x < self.col_num]
        if set(cols) & set(tar_cols) == set(tar_cols):
            score += 1
        for con_col in self.clause_cols['WHERE'] | self.clause_cols['HAVING']:
            if con_col in self.val_hit:
                score += 1
        return score

    def calc_penalty(self, oracle=False):
        if self.len == 0:
            return 0
        if self.bracket > 1:
            return INF
        if self.parent.tok_type == 'col':
            assert self.tok_id >= len(ALL_KWS)
            if self.parent.tasks[-1][0] not in ['WHERE', 'HAVING'] and self.tok_id - len(ALL_KWS) in \
                    self.parent.tasks[-1][1]:
                return INF
            if self.parent.tasks[-1][0] in ['GROUP_BY', 'WHERE'] and self.tok_id == len(ALL_KWS):
                return INF
            if self.tok_id - len(ALL_KWS) in self.fob_cols:
                return INF
            if self.bracket > 0 and self.tasks[-1][0] == 'SELECT':
                op = self.parent.parent.parent
                if op.tok_id == ALL_KWS_MAP['in']:
                    before = op.parent
                    if before.tok_id == ALL_KWS_MAP['not']:
                        before = before.parent
                    assert before.tok_id in range(len(ALL_KWS), len(ALL_KWS) + self.col_num)
                    if (before.tok_id - len(ALL_KWS), self.tok_id - len(ALL_KWS)) not in self.key_pair \
                            and (self.tok_id - len(ALL_KWS), before.tok_id - len(ALL_KWS)) not in self.key_pair:
                        return INF

        elif self.parent.tok_type == 'tbl':
            assert self.tok_id >= len(ALL_KWS) + self.col_num
            if self.tok_id - len(ALL_KWS) - self.col_num in self.fob_tbls:
                return INF
        else:
            if oracle and self.tok_id == ALL_KWS_MAP['value'] and ALL_KWS[self.parent.tok_id] in OPS[1:8] and \
                    self.tasks[-1][
                        0] == 'WHERE':
                if self.parent.parent.tok_id - len(ALL_KWS) not in self.val_hit:
                    return INF
            if self.tok_id == ALL_KWS_MAP['in']:
                cur = self.parent
                if cur.tok_id == ALL_KWS_MAP['not']:
                    cur = cur.parent
                assert cur.tok_id in range(len(ALL_KWS), len(ALL_KWS) + self.col_num)
                if sum(self.key_info[cur.tok_id - len(ALL_KWS)]) == 0:
                    return INF
        return 0
