#!/bin/bash
models=('Transformer' 'GRU' 'LSTM')


for model in "${models[@]}"
do
    for ((max_len=1000;max_len<=10000;epochs+=1000));
    do
        echo "sequence classification with fixed length"
        python create_sequence_classification_data.py --seq_len $max_len
        for ((epochs=100;epochs<=1000;epochs+=100));
        do
                for ((nlayer=2;nlayer<=5;nlayer++));
                do  
                    nhead=$(( 2*nlayer )) 
                    nhid=$(( 2*nhead )) 
                    echo "#########################################################"
                    echo "model: $model, nlayer: $nlayer, nhid: $nhid nhead: $nhead"
                    python main.py --model "$model" --nlayers $nlayer --nhid $nhid --nhead $nhead --log-interval 16000 --epochs $epochs --data_dir "data/sequence_classification/fixed_length"
                done
            
        done

        echo "sequence classification with variable length"
        python create_sequence_classification_data.py --seq_len $max_len --variable_length
        for ((epochs=100;epochs<=1000;epochs+=100));
        do
                for ((nlayer=2;nlayer<=5;nlayer++));
                do  
                    nhead=$(( 2*nlayer )) 
                    nhid=$(( 2*nhead )) 
                    echo "#########################################################"
                    echo "model: $model, nlayer: $nlayer, nhid: $nhid nhead: $nhead"
                    python main.py --model "$model" --nlayers $nlayer --nhid $nhid --nhead $nhead --log-interval 16000 --epochs $epochs --data_dir "data/sequence_classification/variable_length"
                done
            
        done
    done

    for ((max_len=1000;max_len<=10000;epochs+=1000));
    do
        echo "sequence learning with fixed length and fixed increment"
        python create_sequence_learning_data.py --seq_len $max_len
        for ((epochs=100;epochs<=1000;epochs+=100));
        do
                for ((nlayer=2;nlayer<=5;nlayer++));
                do  
                    nhead=$(( 2*nlayer )) 
                    nhid=$(( 2*nhead )) 
                    echo "#########################################################"
                    echo "model: $model, nlayer: $nlayer, nhid: $nhid nhead: $nhead"
                    python main.py --model "$model" --nlayers $nlayer --nhid $nhid --nhead $nhead --log-interval 16000 --epochs $epochs --data_dir "data/sequence_learning/fixed_length/fixed_increment"
                done
            
        done

        echo "sequence learning with fixed length and variable increment"
        python create_sequence_classification_data.py --seq_len $max_len --variable_increment
        for ((epochs=100;epochs<=1000;epochs+=100));
        do
                for ((nlayer=2;nlayer<=5;nlayer++));
                do  
                    nhead=$(( 2*nlayer )) 
                    nhid=$(( 2*nhead )) 
                    echo "#########################################################"
                    echo "model: $model, nlayer: $nlayer, nhid: $nhid nhead: $nhead"
                    python main.py --model "$model" --nlayers $nlayer --nhid $nhid --nhead $nhead --log-interval 16000 --epochs $epochs --data_dir "data/sequence_learning/fixed_length/variable_increment"
                done
            
        done

        echo "sequence learning with variable length and fixed increment"
        python create_sequence_classification_data.py --seq_len $max_len --variable_length 
        for ((epochs=100;epochs<=1000;epochs+=100));
        do
                for ((nlayer=2;nlayer<=5;nlayer++));
                do  
                    nhead=$(( 2*nlayer )) 
                    nhid=$(( 2*nhead )) 
                    echo "#########################################################"
                    echo "model: $model, nlayer: $nlayer, nhid: $nhid nhead: $nhead"
                    python main.py --model "$model" --nlayers $nlayer --nhid $nhid --nhead $nhead --log-interval 16000 --epochs $epochs --data_dir "data/sequence_learning/variable_length/fixed_increment"
                done
            
        done

        echo "sequence learning with variable length and variable increment"
        python create_sequence_classification_data.py --seq_len $max_len --variable_length --variable_increment
        for ((epochs=100;epochs<=1000;epochs+=100));
        do
                for ((nlayer=2;nlayer<=5;nlayer++));
                do  
                    nhead=$(( 2*nlayer )) 
                    nhid=$(( 2*nhead )) 
                    echo "#########################################################"
                    echo "model: $model, nlayer: $nlayer, nhid: $nhid nhead: $nhead"
                    python main.py --model "$model" --nlayers $nlayer --nhid $nhid --nhead $nhead --log-interval 16000 --epochs $epochs --data_dir "data/sequence_learning/variable_length/variable_increment"
                done
            
        done
    done
done


