#!/usr/bin/env bash

models=('Transformer' 'GRU' 'LSTM')

max_len=${2:-5}
epochs=${3:-10}
nlayer=${4:-2}
nhead=${5:-1}
nhid=${6:-8}

run_experiment() {
    for ((e=$1;e<=$1*10;e+=10));
    do
        for ((hid=$2;hid<=$2*4;hid+=2));
        do  
                    
            echo "model: $3, nlayer: $nlayer, nhid: $hid, nhead: $nhead, epochs: $e"
            save_dir="$5/$model-model_$len-length_$e-epochs_$nlayer-layer_$nhead-head_$hid-hiddendim"
            python main.py --model $3 --nlayers $nlayer --nhid $hid --nhead $nhead --epochs $e --data_dir $4 --save $save_dir
            
        done
            
    done
}


for model in "${models[@]}"
do

    for ((len=max_len;len<=max_len*10;len+=100));
    do
        echo "sequence classification with fixed length $len"

        data_dir="data/sequence_classification/fixed_length"
        output_dir="output/seq_cls/fixed_length"

        python create_sequence_classification_data.py --seq_len $len
        run_experiment $epochs $nhid $model $data_dir $output_dir

        echo "#########################################################"
        echo "sequence classification with variable length maximum $len"

        data_dir="data/sequence_classification/variable_length"
        output_dir="output/seq_cls/variable_length"

        python create_sequence_classification_data.py --seq_len $len --variable_length
        run_experiment$epochs $nhid $model $data_dir $output_dir
    done

    for ((len=max_len;len<=max_len*10;len+=100));
    do
        echo "#########################################################"
        echo "sequence learning with fixed length $len and fixed increment"

        data_dir="data/sequence_learning/fixed_length/fixed_increment"
        output_dir="output/seq_lrn/fixed_length/fixed_inc"

        python create_sequence_learning_data.py --seq_len $len
        run_experiment $epochs $nhid $model $data_dir $output_dir

        echo "#########################################################"
        echo "sequence learning with fixed length $len  and variable increment"

        data_dir="data/sequence_learning/fixed_length/variable_increment"
        output_dir="output/seq_lrn/fixed_length/var_inc"

        python create_sequence_learning_data.py --seq_len $len --variable_increment
        run_experiment $epochs $nhid $model $data_dir $output_dir


        echo "#########################################################"
        echo "sequence learning with variable length maximum $len and fixed increment"

        data_dir="data/sequence_learning/variable_length/fixed_increment"
        output_dir="output/seq_lrn/variable_length/fixed_inc"

        python create_sequence_learning_data.py --seq_len $len --variable_length
        run_experiment $epochs $nhid $model $data_dir $output_dir

        echo "#########################################################"
        echo "sequence learning with variable length maximum $len and variable increment"

        data_dir="data/sequence_learning/variable_length/variable_increment"
        output_dir="output/seq_lrn/variable_length/var_inc"

        python create_sequence_learning_data.py --seq_len $len --variable_increment --variable_length
        run_experiment $epochs $nhid $model $data_dir $output_dir
    done

done
