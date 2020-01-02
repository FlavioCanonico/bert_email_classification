#!/bin/bash

cat > /content/Shell/test.tsv << EOF
guid	text
1	$1
EOF

python /content/bert/run_classifier.py 17 \
--task_name=cola \
--do_predict=true \
--data_dir=/content/Shell/ \
--vocab_file=/content/multi_cased_L-12_H-768_A-12/vocab.txt \
--bert_config_file=/content/multi_cased_L-12_H-768_A-12/bert_config.json \
--init_checkpoint=/content/bert_output/model.ckpt-1907 \
--max_seq_length=192 \
--output_dir=/content/Shell/ \
--do_lower_case=False

python /content/pythonscript_percolab.py