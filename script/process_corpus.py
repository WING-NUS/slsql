from stanfordnlp.server import CoreNLPClient

"""
Client for accessing Stanford CoreNLP in Python
"""

import json

client = CoreNLPClient(annotators=['tokenize', 'ssplit', 'pos', 'lemma'],
                       timeout=120000, memory='4G',
                       start_server=False)


def fix(sen):
    sen = sen.replace("'s ", "<#TEMP#s#> ")
    sen = sen.replace("'t ", "<#TEMP#t#> ")
    sen = sen.replace("''", " <#TEMP#q#> ")
    sen = sen.replace('"', ' " ')
    sen = sen.replace("'", " ' ")
    if sen[-1] in ['?', '.']:
        sen = sen[:-1] + (' ' if sen[-2] != ' ' else '') + sen[-1]
    sen = sen.replace("<#TEMP#t#> ", "'t ")
    sen = sen.replace("<#TEMP#s#> ", "'s ")
    sen = sen.replace("<#TEMP#q#>", "''")
    return sen


def parse(text):
    sentences = client.annotate(text).sentence
    tokens = []
    offset = 0
    sen_pos = []
    for sentence in sentences:
        sen_pos.append(len(tokens))
        cur_toks = sentence.token
        cur_toks = [tok for tok in cur_toks]
        tokens.extend(cur_toks)
        offset += len(cur_toks)
    return tokens, sen_pos


def process_corpus(file_name):
    data_ret = []
    with open('data/' + file_name) as f:
        data = json.load(f)
        for i, sample in enumerate(data):
            que = fix(sample['question'])
            tokens, sen_pos = parse(que)
            lemma = [x.lemma for x in tokens]
            toks = [x.word for x in tokens]
            sample['lemma'] = lemma
            sample['toks'] = toks
            if len(sample['lemma']) != len(sample['ant']):
                raise Exception("id:" +str(i) + " annotation is not compatible with the dataset")
            data_ret.append(sample)
    with open('data/' + file_name, 'wt') as out:
        json.dump(data_ret, out, indent=4, separators=(',', ': '))


if __name__ == '__main__':
    print('preprocess training set')
    process_corpus('slsql_train.json')
    print('preprocess dev set')
    process_corpus('slsql_dev.json')

    print('begin preprocess schemas')
    table_ret = []
    with open('data/tables.json') as f:
        tables = json.load(f)
        for id, table in enumerate(tables):
            column_names_lemma = []
            table_names_lemma = []
            for col in table['column_names']:
                tbl_id, col_name = col
                tokens, _ = parse(col[1])
                lemma = ' '.join([x.lemma for x in tokens])
                column_names_lemma.append([tbl_id, lemma])
            for tbl in table['table_names']:
                tokens, _ = parse(tbl)
                lemma = ' '.join([x.lemma for x in tokens])
                table_names_lemma.append(lemma)
            table['column_names_lemma'] = column_names_lemma
            table['table_names_lemma'] = table_names_lemma
            table_ret.append(table)
    with open('data/processed_tables.json', 'wt') as out:
        json.dump(table_ret, out, indent=4, separators=(',', ': '))
    print('finish processing tables')
