#!/usr/bin/env bash

models=('Transformer' 'GRU' 'LSTM')

max_len=${2:-5}
epochs=${3:-10}
nlayer=${4:-1}
nhead=${5:-1}
nhid=${6:-8}

run_experiment() {
    e=$1
    for ((hid=$2;hid<=$2*4;hid*=2));
    do
        for ((nl=nlayer;nl<=nlayer+10;nl+=5));
        do
            for ((nh=nhead;nh<=nhead*4;nh*=2));
            do
                echo "model: $3, nlayer: $nl, nhid: $hid, nhead: $nh, epochs: $e"
                save_dir="$5/$model-model_$len-length_$e-epochs_$nl-layer_$nh-head_$hid-hiddendim"
                python main.py --model $3 --nlayers $nl --nhid $hid --nhead $nh --epochs $e --data_dir $4 --save $save_dir
            done
        done
    done
}

run_single_experiment() {
    save_dir="$6/$5-model_$7-length_$1-epochs_$3-layer_$4-head_$2-hiddendim"
    python main.py --model $5 --nlayers $3 --nhid $2 --nhead $4 --epochs $1 --data_dir $8 --save $save_dir
}

for model in "${models[@]}"
do
    len=5
    echo "sequence classification with fixed length $len"

    data_dir="data/sequence_classification/fixed_length"
    output_dir="output/seq_cls/fixed_length"

    python create_sequence_classification_data.py --seq_len $len
    run_experiment $epochs $nhid $model $data_dir $output_dir

done

python create_plotly_plots.py

head=1

len=5
echo "#########################################################"
echo "sequence learning with fixed length $len and fixed increment"

data_dir="data/sequence_learning/fixed_length/fixed_increment"
output_dir="output/seq_lrn/fixed_length/fixed_inc"

python create_sequence_learning_data.py --seq_len $len

epochs=50
layer=2

nhid=12
model='Transformer'
run_single_experiment $epochs $nhid $layer $head $model $output_dir $len $data_dir

nhid=14
model='LSTM'
run_single_experiment $epochs $nhid $layer $head $model $output_dir $len $data_dir

nhid=16
model='GRU'
run_single_experiment $epochs $nhid $layer $head $model $output_dir $len $data_dir
