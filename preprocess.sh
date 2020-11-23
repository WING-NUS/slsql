#!/usr/bin/env bash
java -Xmx4G -cp "corenlp/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators "tokenize,ssplit,pos,lemma" -port 9000 -timeout 120000 &
python script/attach_annotation.py && python script/process_corpus.py && python script/gen_sql_seq.py
