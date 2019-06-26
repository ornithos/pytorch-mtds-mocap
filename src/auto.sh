#!/bin/bash
for i in {1..8}
do
    python translate.py --action walking --seq_length_out 64 --style_ix ${i} --use_cpu --load ../experiments/model_${i}_5000_res --sample
done


# ls | sed -n 's/\(predictions_[0-9]\).npz/mv "\1.npz" "\1_k6.npz"/p' | sh
