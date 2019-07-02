#!/bin/bash
for i in {1..8}
do
    echo "=============="
    echo $i
    echo "=============="
    #python translate.py --action walking --seq_length_out 64 --iterations 50000  --style_ix $i --learning_rate 0.02 --test_every 1000 --learning_rate_step 2000
    #python translate_legs.py --action walking --seq_length_out 64 --iterations 50000  --style_ix $i --learning_rate 0.02 --test_every 1000 --learning_rate_step 2000
    python learn_mtmodel.py --seq_length_out 64 --input_size 44 --decoder_size 256 --latent_k 3 --batch_size 16 --iterations 10000 --style_ix $i --use_cpu --learning_rate 1e-4 --optimiser Adam --test_every 2000 --learning_rate_step 2000
    #python translate.py --action walking --seq_length_out 64 --iterations 20000  --style_ix $i --learning_rate 0.005 --learning_rate_step 1000
done
