#!/usr/bin/env bash
set -e .

python tftokenizers/download.py -n bert-base-uncased
python tftokenizers/download.py -n bert-base-cased
python tftokenizers/download.py -n bert-base-multilingual-cased
python tftokenizers/download.py -n bert-base-multilingual-uncased
python tftokenizers/download.py -n albert-base-v2
python tftokenizers/download.py -n bert-base-chinese
python tftokenizers/download.py -n distilbert-base-cased
python tftokenizers/download.py -n distilbert-base-multilingual-cased
python tftokenizers/download.py -n KB/bert-base-swedish-cased

# python tftokenizers/download.py -n ctrl
# python tftokenizers/download.py -n t5-base
# python tftokenizers/download.py -n google/t5-v1_1-base
# python tftokenizers/download.py -n google/byt5-small
# python tftokenizers/download.py -n allenai/unifiedqa-t5-base

# python tftokenizers/download.py -n roberta-base
# python tftokenizers/download.py -n allenai/scibert_scivocab_uncased
# python tftokenizers/download.py -n xlm-roberta-base
# python tftokenizers/download.py -n xlm-mlm-xnli15-1024
# python tftokenizers/download.py -n xlm-mlm-tlm-xnli15-1024
#python tftokenizers/download.py -n microsoft/deberta-v2-xlarge
python tftokenizers/download.py -n allenai/longformer-base-4096

# python tftokenizers/download.py -n xlnet-base-cased
# python tftokenizers/download.py -n transfo-xl-wt103
# python tftokenizers/download.py -n microsoft/prophetnet-large-uncased
# python tftokenizers/download.py -n sentence-tftokenizers/all-MiniLM-L6-v2
# python tftokenizers/download.py -n sentence-tftokenizers/paraphrase-distilroberta-base-v2
# python tftokenizers/download.py -n sentence-tftokenizers/all-mpnet-base-v2
# python tftokenizers/download.py -n allenai/led-base-16384
