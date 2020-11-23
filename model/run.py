# The code was initially based on some tutorial scripts from https://github.com/huggingface/transformers
import argparse
import json
import logging
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from evaluation import evaluate_dev
from pytorch_pretrained_bert import BertTokenizer, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.optimization import BertAdam
from slsql import SLSQL
from spider_dataset import edit_sql, write_sql, feature_name, feature_list, SpiderDataset, make_features
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def eval_linking(eg, ef, cp, vp):
    col_num, tbl_num = eg.extra['col_num'], eg.extra['tbl_num']
    ct_linking = ef.labels['ct_linking']

    cp = cp.view(len(ct_linking), -1)
    cp = F.softmax(cp, dim=1)
    vp = vp.view(len(ct_linking), -1)
    vp = F.softmax(vp, dim=1)
    choices = []
    assert len(cp) == len(vp)
    for id, (ehp, etp) in enumerate(zip(cp, vp)):
        which_item = torch.argmax(ehp)
        which_item = which_item.item()
        if which_item < col_num:
            choices.append((which_item, 'col'))
        elif which_item == col_num + tbl_num:
            which_val_col = torch.argmax(etp)
            which_val_col = which_val_col.item()
            if which_val_col < col_num:
                choices.append((which_val_col, 'val'))
            else:
                choices.append(None)
        else:
            choices.append((which_item, 'tbl'))

    true_positive = defaultdict(int)
    all_positive = defaultdict(int)
    actual_positive = defaultdict(int)
    for pred, gold_ct_link, gold_val_link in zip(choices, eg.labels['ct_linking'], eg.labels['val_linking']):
        if gold_ct_link != tbl_num + col_num or (0 <= gold_val_link < col_num):
            gold_type = 'col' if gold_ct_link < col_num else 'tbl'
            if 0 <= gold_val_link < col_num:
                gold_type = 'val'
            actual_positive[gold_type] += 1
            if pred is not None:
                if (pred[1] == gold_type == 'val' and (pred[0] == gold_val_link)) or \
                        (pred[1] == gold_type and pred[0] == gold_ct_link):
                    true_positive[gold_type] += 1

        if pred is not None:
            all_positive[pred[1]] += 1

    return {"true": true_positive, "all": all_positive, "actual": actual_positive, }, choices


def eval_sql(eg, tp, ap):
    col_num, tbl_num = eg.extra['col_num'], eg.extra['tbl_num']
    db = eg.extra['db']
    assert len(tp) == len(ap)

    # need to do some post editing, otherwise some equivalent queries will be judged wrong
    _, seq_toks, aggs, _ = edit_sql(0, tp, ap, db, col_num, tbl_num)
    return write_sql(seq_toks, aggs, col_num, db, eg)


def eval_one(eg, ef, tp, ap, cp, vp):
    link_ret = eval_linking(eg, ef, cp, vp)
    sql = eval_sql(eg, tp, ap)
    return link_ret[0], (link_ret[1], sql)


def metric(true, all, actual):
    precision = true / all if all != 0 else 0
    recall = true / actual if actual != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    return precision, recall, f1


def do_eval(eval_examples, eval_features, tok_preds, agg_preds, ct_link_preds, val_link_preds):
    pred_linking = []
    pred_sqls = []

    true_positive = defaultdict(int)
    all_positive = defaultdict(int)
    actual_positive = defaultdict(int)
    for i, (eg, ef, tp, ap) in enumerate(zip(eval_examples, eval_features, tok_preds, agg_preds)):
        if args.base:
            pred_sql = eval_sql(eg, tp, ap)
        else:
            cp = ct_link_preds[i]
            vp = val_link_preds[i]
            eval_result, (pred_link, pred_sql) = eval_one(eg, ef, tp, ap, cp, vp)

            for t in ['val', 'tbl', 'col']:
                true_positive[t] += eval_result['true'][t]
                all_positive[t] += eval_result['all'][t]
                actual_positive[t] += eval_result['actual'][t]
            pred_linking.append(pred_link)
        pred_sqls.append(pred_sql)

    link_scores = {}
    if not args.base:
        col_p, col_r, col_f = metric(true_positive['col'], all_positive['col'], actual_positive['col'])
        tbl_p, tbl_r, tbl_f = metric(true_positive['tbl'], all_positive['tbl'], actual_positive['tbl'])
        val_p, val_r, val_f = metric(true_positive['val'], all_positive['val'], actual_positive['val'])

        link_scores['col_p'] = col_p
        link_scores['col_r'] = col_r
        link_scores['col_f1'] = col_f

        link_scores['tbl_p'] = tbl_p
        link_scores['tbl_r'] = tbl_r
        link_scores['tbl_f1'] = tbl_f

        link_scores['val_p'] = val_p
        link_scores['val_r'] = val_r
        link_scores['val_f1'] = val_f
    return link_scores, pred_linking, pred_sqls


def eval(eval_examples, eval_features, eval_dataloader):
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    tok_preds = []
    agg_preds = []
    ct_link_preds = []
    val_link_preds = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            (tok_scores, agg_scores, ct_link_scores, val_link_scores) = model(*batch)

        tok_preds += tok_scores
        agg_preds += agg_scores
        ct_link_preds += ct_link_scores
        val_link_preds += val_link_scores
    scores, pred_links, pred_sqls = do_eval(eval_examples, eval_features, tok_preds, agg_preds,
                                            ct_link_preds,
                                            val_link_preds)
    loss = tr_loss / nb_tr_steps if args.do_train else None
    sql_loss = tr_sql_loss / nb_tr_steps if args.do_train else None
    if not args.base:
        linking_loss = tr_linking_loss / nb_tr_steps if args.do_train else None
        scores['linking_loss'] = linking_loss
    scores['acc'] = evaluate_dev(pred_sqls)
    scores['loss'] = loss
    scores['sql_loss'] = sql_loss

    return scores, pred_links, pred_sqls


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default='data/', type=str)
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str)
    parser.add_argument("--output_dir", default='output', type=str)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--hard", action='store_true')
    parser.add_argument("--base", action='store_true')
    parser.add_argument("--oracle", action='store_true')
    parser.add_argument("--do_lower_case", action='store_true', default=True)
    parser.add_argument("--train_batch_size", default=4, type=int)
    parser.add_argument("--eval_batch_size", default=20, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float, )
    parser.add_argument("--num_train_epochs", default=30, type=float)
    parser.add_argument("--beam_width", default=4, type=int)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_from_epoch', type=int, default=15)
    parser.add_argument("--save_all_models", action='store_true', default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    if args.hard and args.base:
        raise ValueError("Hard and base cannot both be set")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    dataset = SpiderDataset()

    model = SLSQL.from_pretrained(args.bert_model)
    model.args = args
    model.to(device)

    if args.do_train and args.bert_model == 'bert-base-uncased':
        make_features(tokenizer)
    else:
        for x in [feature_name(x) for x in feature_list]:
            tokenizer.basic_tokenizer.never_split += (x,)

    eval_examples = dataset.get_dev_examples(args.data_dir)
    eval_features = dataset.convert_examples_to_features(eval_examples, 512, tokenizer, mode='dev')

    max_item_len = max([max([len(x) for x in f.labels['item_pos']]) for f in eval_features])
    all_item_pos = []
    for each in eval_features:
        cur_item_pos = each.labels['item_pos'] + [[-100] * max_item_len]
        now = pad_sequence([torch.tensor(x) for x in cur_item_pos], batch_first=True, padding_value=-100)
        all_item_pos.append(now[:-1])
    all_item_pos = pad_sequence(all_item_pos, batch_first=True, padding_value=-100)
    all_item_pos = pad_sequence(all_item_pos, batch_first=True)

    eval_data = TensorDataset(torch.tensor([f.input_ids for f in eval_features], dtype=torch.long),
                              torch.tensor([f.input_mask for f in eval_features], dtype=torch.long),
                              torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long),
                              pad_sequence([torch.tensor(f.labels['que_tok_pos']) for f in eval_features],
                                           batch_first=True),
                              all_item_pos,
                              pad_sequence([torch.tensor(f.labels['belongs'], dtype=torch.long) for f in eval_features],
                                           batch_first=True),
                              pad_sequence(
                                  [torch.tensor(f.labels['key_info'], dtype=torch.float) for f in eval_features],
                                  batch_first=True),
                              pad_sequence(
                                  [torch.tensor(f.labels['key_pairs'], dtype=torch.long) for f in eval_features],
                                  batch_first=True, padding_value=-100),
                              torch.tensor([len(f.labels['que_tok_pos']) for f in eval_features], dtype=torch.long),
                              torch.tensor([f.labels['col_num'] for f in eval_features], dtype=torch.long),
                              torch.tensor([f.labels['tbl_num'] for f in eval_features], dtype=torch.long),
                              pad_sequence([torch.tensor(f.labels['ct_linking']) for f in eval_features],
                                           batch_first=True, padding_value=-100),
                              pad_sequence([torch.tensor(f.labels['val_linking']) for f in eval_features],
                                           batch_first=True, padding_value=-100))

    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    if args.do_train:

        num_train_optimization_steps = None
        train_examples = dataset.get_train_examples(args.data_dir)
        train_features = dataset.convert_examples_to_features(train_examples, 512, tokenizer)
        logger.info("***** Begin training *****")
        max_item_len = max([max([len(x) for x in f.labels['item_pos']]) for f in train_features])

        all_item_pos = []
        for each in train_features:
            cur_item_pos = each.labels['item_pos'] + [[-100] * max_item_len]
            now = pad_sequence([torch.tensor(x) for x in cur_item_pos], batch_first=True, padding_value=-100)
            all_item_pos.append(now[:-1])
        all_item_pos = pad_sequence(all_item_pos, batch_first=True, padding_value=-100)
        all_item_pos = pad_sequence(all_item_pos, batch_first=True)

        train_data = TensorDataset(torch.tensor([f.input_ids for f in train_features], dtype=torch.long),
                                   torch.tensor([f.input_mask for f in train_features], dtype=torch.long),
                                   torch.tensor([f.segment_ids for f in train_features], dtype=torch.long),
                                   pad_sequence([torch.tensor(f.labels['que_tok_pos']) for f in train_features],
                                                batch_first=True),
                                   all_item_pos,
                                   pad_sequence(
                                       [torch.tensor(f.labels['belongs'], dtype=torch.long) for f in train_features],
                                       batch_first=True),
                                   pad_sequence(
                                       [torch.tensor(f.labels['key_info'], dtype=torch.float) for f in train_features],
                                       batch_first=True),
                                   pad_sequence(
                                       [torch.tensor(f.labels['key_pairs'], dtype=torch.float) for f in train_features],
                                       batch_first=True, padding_value=-100),
                                   torch.tensor([len(f.labels['que_tok_pos']) for f in train_features],
                                                dtype=torch.long),
                                   torch.tensor([f.labels['col_num'] for f in train_features], dtype=torch.long),
                                   torch.tensor([f.labels['tbl_num'] for f in train_features], dtype=torch.long),
                                   pad_sequence([torch.tensor(f.labels['ct_linking']) for f in train_features],
                                                batch_first=True,
                                                padding_value=-100),
                                   pad_sequence([torch.tensor(f.labels['val_linking']) for f in train_features],
                                                batch_first=True,
                                                padding_value=-100),
                                   pad_sequence([torch.tensor(f.labels['seq']) for f in train_features],
                                                batch_first=True,
                                                padding_value=-100),
                                   pad_sequence([torch.tensor(f.labels['agg']) for f in train_features],
                                                batch_first=True,
                                                padding_value=-100)
                                   )

        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        num_train_optimization_steps = int(len(train_examples) / args.train_batch_size) * args.num_train_epochs

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

        desc_str = "ITERATION - sql_loss: {:.8f}, linking_loss: {:.8f}"
        pbar = tqdm(
            initial=0, leave=False, total=len(train_dataloader),
            desc=desc_str.format(0, 0)
        )

        nb_tr_steps = 0
        tr_loss = 0
        best_acc = 0
        for epoch in range(int(args.num_train_epochs)):
            logger.info("***** begin epoch = %d *****", epoch)
            model.train()
            tr_loss = 0
            tr_sql_loss = 0
            tr_linking_loss = 0
            nb_tr_steps = 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)

                sql_loss, linking_loss = model(*batch)
                loss = sql_loss + linking_loss

                loss.backward()

                pbar.desc = desc_str.format(sql_loss, linking_loss)
                pbar.update(1)

                tr_loss += loss.item()
                tr_sql_loss += sql_loss.item()
                if not args.base:
                    tr_linking_loss += linking_loss.item()
                nb_tr_steps += 1
                optimizer.step()
                optimizer.zero_grad()

            eval_scores, pred_links, pred_sqls = eval(eval_examples, eval_features, eval_dataloader)
            epoch_output_dir = os.path.join(args.output_dir,
                                            'epoch_%d_%.4f_%.4f' % (epoch, eval_scores['loss'], eval_scores['acc']))
            if not os.path.exists(epoch_output_dir):
                os.makedirs(epoch_output_dir)

            if epoch > args.save_from_epoch and (eval_scores['acc'] >= best_acc or args.save_all_models):
                output_model_file = os.path.join(epoch_output_dir, WEIGHTS_NAME)
                output_config_file = os.path.join(epoch_output_dir, CONFIG_NAME)
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(), output_model_file)
                model_to_save.config.to_json_file(output_config_file)
                tokenizer.save_vocabulary(epoch_output_dir)

            with open(os.path.join(epoch_output_dir, "eval_scores.txt"), "w") as writer:
                logger.info("***** Eval scores *****")
                for key in sorted(eval_scores.keys()):
                    logger.info("  %s = %s", key, str(eval_scores[key]))
                    writer.write("%s = %s\n" % (key, str(eval_scores[key])))

            with open(os.path.join(epoch_output_dir, "pred_sql.txt"), 'wt') as out:
                out.writelines([' '.join(pred_sql) + "\n" for pred_sql in pred_sqls])

            if not args.base and not args.oracle:
                pred_details = dataset.print_result(args.data_dir, pred_links, pred_sqls)
                with open(os.path.join(epoch_output_dir, "pred_details.json"), 'wt') as out:
                    json.dump(pred_details, out, sort_keys=True, indent=4, separators=(',', ': '))

            best_acc = max(best_acc, eval_scores['acc'])
            logger.info("***** best acc = %.4f *****", best_acc)
            logger.info("***** end epoch = %d *****", epoch)

    if args.do_eval:
        model.to(device)
        eval_scores, pred_links, pred_sqls = eval(eval_examples, eval_features, eval_dataloader)

        eval_output_dir = os.path.join(args.output_dir, 'eval_%.4f' % (eval_scores['acc']))
        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)

        with open(os.path.join(eval_output_dir, "eval_scores.txt"), "w") as writer:
            logger.info("***** Eval scores *****")
            for key in sorted(eval_scores.keys()):
                logger.info("  %s = %s", key, str(eval_scores[key]))
                writer.write("%s = %s\n" % (key, str(eval_scores[key])))

        with open(os.path.join(eval_output_dir, "eval_scores.json"), 'wt') as out:
            json.dump(eval_scores, out, sort_keys=True, indent=4, separators=(',', ': '))

        with open(os.path.join(eval_output_dir, "pred_sql.txt"), 'wt') as out:
            out.writelines([' '.join(pred_sql) + "\n" for pred_sql in pred_sqls])

        if not args.base and not args.oracle:
            pred_details = dataset.print_result(args.data_dir, pred_links, pred_sqls)
            with open(os.path.join(eval_output_dir, "pred_details.json"), 'wt') as out:
                json.dump(pred_details, out, sort_keys=True, indent=4, separators=(',', ': '))
