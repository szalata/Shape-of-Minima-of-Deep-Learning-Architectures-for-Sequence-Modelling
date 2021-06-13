#!/bin/bash
models=('Transformer' 'LSTM')

for model in "${models[@]}"
do
    for ((nlayer=2;nlayer<=5;nlayer++));
    do
        for ((nhead=2;nhead<=5;nhead+=2));
        do
            
            nhid=$(( 2*nhead )) 
            echo "#########################################################"
            echo "model: $model, nlayer: $nlayer, nhid: $nhid nhead: $nhead"
            python main.py --model "$model" --nlayers $nlayer --nhid $nhid --nhead $nhead --log-interval 16000
        
        done
    done
done