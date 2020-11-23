from copy import deepcopy
from queue import Queue

import torch
import torch.nn.functional as F
from beam import BeamSearchNode
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from spider_dataset import ALL_KWS, ALL_KWS_MAP, TASK_KWS, trans
from torch import nn
from torch.nn import Embedding

INF = 100000000
NINF = -INF


class SLSQL(BertPreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.item_gru = nn.GRU(input_size=config.hidden_size, hidden_size=config.hidden_size, batch_first=True)
        self.que_gru = nn.GRU(input_size=config.hidden_size * 2, hidden_size=config.hidden_size, batch_first=True,
                              bidirectional=True)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.ct_link_scorer = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, 1)
        )
        self.val_link_scorer = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, 1)
        )

        self.item_que_embedding = nn.Sequential(
            nn.Linear(config.hidden_size * 2 + 2, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        self.kw_emb = Embedding(len(ALL_KWS), config.hidden_size)

        self.item_attner = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        self.context_attner = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.que_attner = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        self.kw_scorer = nn.Sequential(nn.Linear(config.hidden_size * 3, config.hidden_size),
                                       nn.Tanh(),
                                       nn.Linear(config.hidden_size, len(ALL_KWS)))
        self.col_scorer = nn.Sequential(nn.Linear(config.hidden_size * 3, config.hidden_size),
                                        nn.Tanh(),
                                        nn.Linear(config.hidden_size, 1))
        self.tbl_scorer = nn.Sequential(nn.Linear(config.hidden_size * 3, config.hidden_size),
                                        nn.Tanh(),
                                        nn.Linear(config.hidden_size, 1))
        self.agg_scorer = nn.Sequential(nn.Linear(config.hidden_size * 2, config.hidden_size),
                                        nn.Tanh(),
                                        nn.Linear(config.hidden_size, 5))
        self.decoder_gru = nn.GRU(input_size=config.hidden_size * 4, hidden_size=config.hidden_size, batch_first=True)

        self.register_buffer("kws_ids", torch.arange(100))

        self.args = None
        self.apply(self.init_bert_weights)

    def get_emb(self, tok_name):
        return self.kw_emb(self.kws_ids[ALL_KWS_MAP[tok_name]])

    def attn_score(self, hidden, targets, attn):
        energy = attn(targets)
        return torch.sum(hidden * energy, dim=-1)

    def item_attn(self, h, item_que_embs):
        attn_energies = self.attn_score(h, item_que_embs, self.item_attner)
        attn_weights = F.softmax(attn_energies, dim=-1).unsqueeze(1)
        attned = attn_weights.bmm(item_que_embs).squeeze(1)
        return attned

    def context_attn(self, h, contexts, fob):
        attn_energies = self.attn_score(h.view(-1), contexts, self.context_attner)
        attn_energies[fob] = NINF
        attn_weights = F.softmax(attn_energies, dim=-1).unsqueeze(0)
        attned = attn_weights.mm(contexts).squeeze(0)
        return attned

    def que_attn(self, h, que):
        attn_energies = self.attn_score(h.view(-1), que, self.que_attner)
        attn_weights = F.softmax(attn_energies, dim=-1).unsqueeze(0)
        attned = attn_weights.mm(que).squeeze(0)
        return attned

    def pred_linking(self, ct_scores, val_scores):
        ct_ret = torch.argmax(ct_scores.squeeze(-1), dim=1)
        val_ret = torch.argmax(val_scores.squeeze(-1), dim=1)
        return ct_ret, val_ret

    def reference(self, que_emb, item_emb, col_num, tbl_num, ct_linking, val_linking):
        if self.args.hard:
            ids = []
            for cl, vl in zip(ct_linking, val_linking):
                cl = cl.item()
                vl = vl.item()
                if cl < col_num + tbl_num:
                    ids.append(cl)
                elif 0 <= vl < col_num:
                    ids.append(vl)
                else:
                    ids.append(col_num + tbl_num)
            refs = item_emb[ids]
            return torch.cat([que_emb, refs], dim=1)
        else:
            refs = []
            for cl, vl in zip(ct_linking, val_linking):
                if torch.argmax(cl, dim=0) >= col_num + tbl_num and torch.argmax(vl, dim=0) < col_num:
                    vl = torch.softmax(vl, dim=0)
                    emb = vl * item_emb
                    emb = torch.mean(emb, dim=0)
                    refs.append(emb)
                else:
                    cl = torch.softmax(cl, dim=0)
                    emb = cl * item_emb
                    emb = torch.mean(emb, dim=0)
                    refs.append(emb)
            refs = torch.stack(refs)
            return torch.cat([que_emb, refs], dim=1)

    def forward(self, input_ids, input_mask, segment_ids, que_tok_pos, item_pos, belongs, key_infos, key_pairs,
                que_tok_num, col_nums, tbl_nums, ct_link_labels, val_link_labels, seqs=None, aggs=None):

        sequence_output, _ = self.bert(input_ids, segment_ids, input_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        all_seq_loss = []

        all_ct_loss = []
        all_val_loss = []

        all_seqs = []
        all_aggs = []

        all_ct_link_scores = []
        all_val_link_scores = []

        for i in range(len(input_ids)):
            que_index = que_tok_pos[i][:que_tok_num[i]]
            que_emb = sequence_output[i, que_index]
            cur_item_pos = item_pos[i]
            col_num = col_nums[i].item()
            tbl_num = tbl_nums[i].item()
            item_emb = []
            for j in range(col_num + tbl_num + 1):
                end = 1
                while end < len(cur_item_pos[j]) and cur_item_pos[j, end] != -100:
                    end += 1
                now_item_pos = cur_item_pos[j, :end]
                hs = sequence_output[i, now_item_pos]
                _, h = self.item_gru(hs.unsqueeze(0))
                h = h.squeeze()
                item_emb.append(h)
            item_emb = torch.stack(item_emb)

            if self.args.base:
                enc_hid = sequence_output[i, 0].reshape(1, 1, -1)
            else:
                item_que_emb_wo_key = []
                for j, item in enumerate(item_emb):
                    cur_emb_wo_key = []
                    for que in que_emb:
                        cur_emb_wo_key.append(torch.cat([item, que]))
                    cur_emb_wo_key = torch.stack(cur_emb_wo_key)
                    item_que_emb_wo_key.append(cur_emb_wo_key)

                item_que_emb_wo_key = torch.stack(item_que_emb_wo_key)
                que_item_emb = item_que_emb_wo_key.transpose(0, 1).squeeze()

                ct_link_scores = self.ct_link_scorer(que_item_emb)
                all_ct_link_scores.append(ct_link_scores)
                val_link_scores = self.val_link_scorer(que_item_emb)
                val_link_scores[:, col_num:col_num + tbl_num] = NINF
                all_val_link_scores.append(val_link_scores)

                if self.training or self.args.oracle:
                    pred_ct_link = ct_link_labels[i][:que_tok_num[i]]
                    pred_val_link = val_link_labels[i][:que_tok_num[i]]
                else:
                    pred_ct_link, pred_val_link = self.pred_linking(ct_link_scores, val_link_scores)
                if self.args.hard:
                    combine_input = self.reference(que_emb, item_emb, col_num, tbl_num, pred_ct_link, pred_val_link)
                else:
                    combine_input = self.reference(que_emb, item_emb, col_num, tbl_num, ct_link_scores, val_link_scores)
                que_emb, enc_hid = self.encode_que(combine_input)

            item_que_emb_w_key = []
            for j, item in enumerate(item_emb[:-1]):
                cur_emb_w_key = []
                if j < col_num:
                    key = key_infos[i, j]
                else:
                    key = torch.tensor([0, 0], dtype=torch.float, device=que_emb.device)
                for k, que in enumerate(que_emb):
                    cur_emb_w_key.append(torch.cat([item, que, key]))
                cur_emb_w_key = torch.stack(cur_emb_w_key)
                item_que_emb_w_key.append(cur_emb_w_key)
            item_que_emb_w_key = torch.stack(item_que_emb_w_key)
            item_que_emb_w_key = self.item_que_embedding(item_que_emb_w_key)

            cur_belongs = [x.item() for x in belongs[i, :col_num]]

            ct_hits = set()
            val_hits = set()
            if not self.args.base:
                ct_hits = set([x.item() for x in pred_ct_link if 0 <= x < col_num + tbl_num])
                val_hits = set([x.item() for x in pred_val_link if 0 <= x < col_num + tbl_num])

            key_pair = [(x[0].item(), x[1].item()) for x in key_pairs[i] if x[0].item() != -100]

            if self.training:
                cur_loss = self.decode(1, col_num, tbl_num, enc_hid, que_emb, item_emb, item_que_emb_w_key,
                                         cur_belongs, key_pair, key_infos[i], ct_hits, val_hits, seqs[i], aggs[i])
                all_seq_loss.append(cur_loss)
                if self.args.base:
                    all_ct_loss.append(0)
                    all_val_loss.append(0)
                else:
                    all_ct_loss.append(F.cross_entropy(ct_link_scores, pred_ct_link.unsqueeze(-1)))
                    all_val_loss.append(F.cross_entropy(val_link_scores, pred_val_link.unsqueeze(-1)))
            else:
                decode_ret = self.decode(1 if self.training else self.args.beam_width, col_num, tbl_num,
                                         enc_hid, que_emb, item_emb, item_que_emb_w_key,
                                         cur_belongs, key_pair, key_infos[i], ct_hits, val_hits)
                cur_seq, cur_agg = decode_ret
                all_seqs.append(cur_seq)
                all_aggs.append(cur_agg)

        if self.training:
            return sum(all_seq_loss), sum(all_ct_loss) + sum(all_val_loss)
        else:
            return all_seqs, all_aggs, all_ct_link_scores, all_val_link_scores

    def encode_que(self, ques):
        o, h = self.que_gru(ques.unsqueeze(0))
        ques = o[:, :, :self.config.hidden_size] + o[:, :, self.config.hidden_size:]
        hid = h[0, :, :self.config.hidden_size] + h[1, :, :self.config.hidden_size:]
        return ques.squeeze(0), hid.unsqueeze(0)

    def decode(self, beam_width, col_num, tbl_num, enc_hid, que_emb, item_emb, item_que_emb, belongs, key_pair,
               key_info, ct_hits, val_hits, seq_gold=None, agg_gold=None):
        cur_task = 'SELECT'
        tasks = [(cur_task, set())]
        cur_range, cur_range_type = trans('SELECT', 'SELECT', col_num, tbl_num)
        start = BeamSearchNode(col_num, tbl_num, None, enc_hid, ALL_KWS_MAP['SELECT'],
                               0, 0, 0, cur_range, cur_range_type, tasks, None, None, [], [])

        q = Queue()
        q.put(start)

        end_nodes = []
        while not q.empty():
            candidates = []
            for _ in range(q.qsize()):
                node = q.get()

                if node.tok_id == ALL_KWS_MAP['EOS'] or node.len >= 45:
                    end_nodes.append(node)
                    continue

                new_nodes = self.decode_node(node, beam_width, col_num, tbl_num, que_emb, item_emb, item_que_emb,
                                             belongs,
                                             key_pair, key_info,
                                             ct_hits, val_hits, seq_gold)

                candidates += new_nodes

            candidates = sorted(candidates, key=lambda x: -x.eval(self.args.oracle))
            length = min(len(candidates), beam_width)
            for i in range(length):
                q.put(candidates[i])
        end_nodes = sorted(end_nodes, key=lambda x: -x.eval(self.args.oracle, True))
        end_nodes = end_nodes[:min(len(end_nodes), beam_width)]

        if seq_gold is not None:
            assert len(end_nodes) == 1
            end_node = end_nodes[0]
            seq_gold = seq_gold[1:end_node.len + 1]
            agg_gold = agg_gold[1:end_node.len + 1].float()
            all_tok_scores, agg_rets, agg_masks = end_node.recover_scores()
            assert len(seq_gold) == len(agg_gold) == len(all_tok_scores) == len(agg_rets)
            all_agg_scores = []
            for id in agg_masks:
                agg_input = agg_rets[id]
                assert agg_input is not None
                all_agg_scores.append(self.agg_scorer(agg_input))

            agg_masks = torch.tensor(agg_masks, device=all_tok_scores[0].device)

            all_tok_scores = torch.stack(all_tok_scores)
            all_agg_scores = torch.stack(all_agg_scores)
            all_agg_logits = torch.sigmoid(all_agg_scores)

            seq_loss = F.cross_entropy(all_tok_scores, seq_gold)

            agg_gold = agg_gold.view(-1, 5)[agg_masks]
            agg_loss = F.binary_cross_entropy(all_agg_logits, agg_gold)

            return seq_loss + agg_loss
        else:
            end_node=end_nodes[0]
            seq_ret, agg_inputs = end_node.recover_ret()
            agg_ret = [x if x is None else self.agg_scorer(x) for x in agg_inputs]
            agg_ret = [x if x is None else torch.sigmoid(x) for x in agg_ret]
            return seq_ret, agg_ret

    def decode_node(self, node, beam_width, col_num, tbl_num, que_emb, item_embs, item_que_embs, bels, key_pair,
                    key_info, ct_hits, val_hits, seq_gold=None):
        h = node.h
        tok_range = node.tok_range
        tok_type = node.tok_type
        bracket = node.bracket
        cur_task_name = node.tasks[-1][0]

        fob_cols = []
        fob_tbls = []
        if not self.training and not self.args.base:
            col_inf_tbls = [bels[x] for x in val_hits if x >= 1] + [bels[x] for x in ct_hits if 1 <= x < col_num]
            for x in range(1, col_num):
                if x not in ct_hits and x not in val_hits and bels[x] + col_num not in ct_hits and bels[
                    x] not in col_inf_tbls:
                    fob_cols.append(x)
            for x in range(tbl_num):
                if x not in col_inf_tbls and x + col_num not in ct_hits:
                    fob_tbls.append(x)

        item_attns = self.item_attn(self.get_emb(cur_task_name), item_que_embs)
        task_fobs = list(node.tasks[-1][1]) if cur_task_name not in ['WHERE', 'HAVING'] else []
        ctx_fobs = list(set(task_fobs + fob_cols + [x + col_num for x in fob_tbls]))
        ctx_attn = self.context_attn(h, item_attns, ctx_fobs)
        q_attn = self.que_attn(h, que_emb)
        all_col_tbl = torch.cat(
            [h.view(-1).repeat(col_num + tbl_num, 1), item_attns, q_attn.repeat(col_num + tbl_num, 1)], dim=1)
        default_score = [NINF] * (len(ALL_KWS) + col_num + tbl_num)
        cur_tok_scores = torch.tensor(default_score, dtype=torch.float, device=h.device)

        tok_scores = self.col_scorer(all_col_tbl[:col_num]).squeeze(1) if tok_type == 'col' \
            else self.tbl_scorer(all_col_tbl[col_num:]).squeeze(1) if tok_type == 'tbl' \
            else self.kw_scorer(torch.cat([h.view(-1), ctx_attn, q_attn]))

        ret = []
        for pick in range(beam_width):
            next_tasks = deepcopy(node.tasks)
            new_bracket = bracket
            if tok_type == 'col':
                if not self.training:
                    if cur_task_name not in ['WHERE', 'HAVING'] or node.seq_ret(0) == ALL_KWS_MAP['and']:
                        tok_scores[list(next_tasks[-1][1])] = NINF
                    if cur_task_name in ['GROUP_BY', 'WHERE']:
                        tok_scores[0] = NINF
                    tok_scores[fob_cols] = NINF
                if pick >= col_num:
                    break
                cur_choice = torch.sort(tok_scores, descending=True)[1][pick]
                cur_prob = torch.log_softmax(tok_scores, dim=0)[cur_choice]
                cur_choice += len(ALL_KWS)
                cur_choice = cur_choice.item()
                cur_tok_scores[len(ALL_KWS):len(ALL_KWS) + col_num] = tok_scores
            elif tok_type == 'tbl':
                if not self.training:
                    tok_scores[fob_tbls] = NINF
                    assert next_tasks[-2][0] == 'SELECT'

                    sel_tbls = [bels[x] for x in next_tasks[-2][1] if x != 0]
                    if len(sel_tbls):
                        tok_scores[sel_tbls] = INF

                    mask_range = [x - col_num for x in list(next_tasks[-1][1])]
                    tok_scores[mask_range] = NINF

                if pick >= tbl_num:
                    break
                cur_choice = torch.sort(tok_scores, descending=True)[1][pick]
                cur_prob = torch.log_softmax(tok_scores, dim=0)[cur_choice]
                cur_choice += len(ALL_KWS) + col_num
                cur_choice = cur_choice.item()
                cur_tok_scores[len(ALL_KWS) + col_num:] = tok_scores
            else:
                if pick >= len(tok_range):
                    break
                cur_choice = torch.sort(tok_scores[tok_range], descending=True)[1][pick]
                cur_prob = torch.log_softmax(tok_scores[tok_range], dim=0)[cur_choice]
                cur_choice = tok_range[cur_choice]
                cur_tok_scores[tok_range] = tok_scores[tok_range]

            assert cur_choice is not None and cur_choice in tok_range
            if self.training:
                tok_id = seq_gold[node.len + 1].item()
            else:
                tok_id = cur_choice

            assert tok_id in tok_range

            agg_input = None
            if tok_type == 'col' and cur_task_name in ['SELECT', 'HAVING', 'ORDER_BY']:
                agg_mask = node.len
                to_agg = item_attns[tok_id - len(ALL_KWS)]
                agg_input = torch.cat([h.view(-1), to_agg])
            else:
                agg_mask = None

            if tok_id < len(ALL_KWS):
                last_emb = self.get_emb(ALL_KWS[tok_id])
                if next_tasks[-1][0] in ['WHERE', 'HAVING'] and tok_id == ALL_KWS_MAP['value'] and node.seq_ret(0) == \
                        ALL_KWS_MAP['=']:
                    next_tasks[-1][1].add(node.seq_ret(1) - len(ALL_KWS))
            if tok_id >= len(ALL_KWS):
                last_emb = item_embs[tok_id - len(ALL_KWS)]
                if next_tasks[-1][0] not in ['WHERE', 'HAVING']:
                    next_tasks[-1][1].add(tok_id - len(ALL_KWS))
            elif tok_id == ALL_KWS_MAP['(']:
                new_bracket += 1
            elif tok_id == ALL_KWS_MAP[')']:
                new_bracket -= 1
                original = next_tasks
                while len(next_tasks) and next_tasks[-1][0] != 'SELECT':
                    next_tasks = next_tasks[:-1]
                next_tasks = next_tasks[:-1]
                if not len(next_tasks):
                    raise Exception(original)
            tok_type = ALL_KWS[tok_id] if tok_id < len(ALL_KWS) else 'col' if tok_id < col_num + len(ALL_KWS) else 'tbl'
            if tok_id < len(ALL_KWS) and ALL_KWS[tok_id] in TASK_KWS:
                next_tasks.append((ALL_KWS[tok_id], set()))
            new_task_name = next_tasks[-1][0]
            next_tok_range, next_tok_type = trans(new_task_name, tok_type, col_num, tbl_num, has_left=new_bracket > 0)
            cur_task_emb = self.get_emb(new_task_name)
            if new_bracket > 0:
                cur_task_emb = torch.mean(torch.stack([cur_task_emb, self.get_emb('(')], dim=0), dim=0)

            input = torch.cat([cur_task_emb, last_emb, ctx_attn, q_attn])
            input = input.reshape(1, 1, -1)
            _, next_h = self.decoder_gru(input, h)

            ret.append(BeamSearchNode(col_num, tbl_num, node, next_h, tok_id, new_bracket, cur_prob, node.len + 1,
                                      next_tok_range, next_tok_type, next_tasks, cur_tok_scores, agg_mask,
                                      fob_cols, fob_tbls, ct_hit=ct_hits, val_hit=val_hits,
                                      agg_input=agg_input, key_pair=key_pair, key_info=key_info))
        return ret
