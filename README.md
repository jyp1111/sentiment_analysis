# sentiment_analysis
## Datasets
한국어 스팀 리뷰 감성 분류 데이터 (100K)
## Run Codes
### pretraining
python run_pretraining.py --input_file document file path
### fine tuning
python run_classifier.py --do_train
### evaluation
python run_classifier.py --do_eval --init_checkpoint checkpoint path
## Reference
https://arxiv.org/abs/1905.05583
