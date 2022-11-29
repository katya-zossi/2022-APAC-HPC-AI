#!/bin/bash

# change the path to your own directory
path="/scratch/wo64/ez3336/2022-APAC-HPC-AI/Deep_Learning_Based_DNA_Sequence_Fast_Decoding"
output_path="/scratch/wo64/ez3336/data/multi_gpus"

## To train a single model with two gpus
horovodrun -np 2 --timeline-filename $output_path/cnn_timeline.json python3 $path/deep_tf.py -m cnn

## To train multiple models
# array=( cnn unet se_cnn )
# for i in "${array[@]}"
# do
# 	horovodrun -np 2 --timeline-filename $output_path/"$i"_timeline.json python3 "$path"/deep_tf.py -m $i
# done
