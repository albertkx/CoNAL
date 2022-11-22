#!/bin/sh

.PHONY: data

data:
	python3 -m preprocessing.make_data \
		--out_path data/  \
		--tacred_data_path /path/to/tacred/

noise_analysis_splits:
	python3 -m preprocessing.make_noise_splits  \
		--out_path noise_splits_nottrain/ \
		--data_path data/

train_baseline:
	CUDA_VISIBLE_DEVICES=0 python3 -m training.train_classifier \
		data/trec10/0/ \
		models/trec10/0/bert_baseline_seed_1/ \
		--ood_training_type vanilla \
		--model_type bert_base \
		--num_training_steps 5000 \
		--checkpoint_steps 100 \
		--seed 1 \
		--retrain_if_exists \
		--do_loss_logging

train_ccl_wikitext:
	CUDA_VISIBLE_DEVICES=0 python3 -m training.train_classifier \
		data/trec10/0/ \
		models/trec10/0/bert_ccl_seed_1/ \
		--ood_training_type ccl \
		--ood_path data/wikitext/ \
		--model_type bert_base \
		--num_training_steps 5000 \
		--checkpoint_steps 100 \
		--seed 1 \
		--retrain_if_exists \
		--do_loss_logging

train_oe_wikitext:
	CUDA_VISIBLE_DEVICES=0 python3 -m training.train_classifier \
		data/trec10/0/ \
		models/trec10/0/bert_oe_seed_1/ \
		--ood_training_type oe \
		--ood_path data/wikitext/ \
		--model_type bert_base \
		--num_training_steps 5000 \
		--checkpoint_steps 100 \
		--seed 1 \
		--retrain_if_exists \
		--do_loss_logging

generate:
	CUDA_VISIBLE_DEIVCES=0 python3 -m ood_generation.generate_np \
		--dataset trec10 \
		--gold_ood_class 0 \
		--example_generator_size large \
		--label_generator_size gpt3 \
		--num_generations 100000 \
		--generation_batch_size 16 \
		--label_generation_iterations 100 \
		--output_dir_name generations_test \
		--do_moby \
		--seed 1; \

score_cnl:
	CUDA_VISIBLE_DEVICES=0 python3 -m scoring.score bert_ccl trec10
