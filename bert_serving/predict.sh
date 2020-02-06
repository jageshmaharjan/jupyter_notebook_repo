export BERT_BASE_DIR='chinese_L-12_H-768_A-12'
python3 run_classifier.py \
		--data_dir=data \
		--task_name=chinese \
		--vocab_file=$BERT_BASE_DIR/vocab.txt \
		--bert_config_file=$BERT_BASE_DIR/bert_config.json \
		--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
		--max_seq_length=256 \
		--train_batch_size=16 \
		--learning_rate=2e-5 \
		--num_train_epochs=30 \
		--output_dir=chinese_model \
		--use_tpu=False \
		--do_predict=true
