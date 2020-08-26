#!/bin/sh

a=("$1")
b=("$2")
c=("$3")

python model_test.py $a $b
python bleu_eval.py $c