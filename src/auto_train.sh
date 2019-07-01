#!/bin/bash
for i in {2..8}
do
    echo "=============="
    echo $i
    echo "=============="
    python translate.py --action walking --seq_length_out 64 --iterations 50000  --style_ix $i --learning_rate 0.02 --test_every 1000 --learning_rate_step 2000
    #python translate.py --action walking --seq_length_out 64 --iterations 20000  --style_ix $i --learning_rate 0.005 --learning_rate_step 1000
done
