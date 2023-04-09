#python3 run_squad.py --model_type bert --model_name_or_path bert-base-uncased --output_dir out

export SQUAD_DIR=./squad_datasets

python run_squad.py \
  --model_type bert \
  --model_name_or_path ./squad_datasets/debug_squad/checkpoint-7000 \
  --do_eval \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir $SQUAD_DIR/debug_squad/
