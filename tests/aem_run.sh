# train the AEM  model
cd ..

python -m tests.main_v2 --data_dir=amazon_cellphone_index_dataset/seq_min_count5/ --input_train_dir=amazon_cellphone_index_dataset/seq_min_count5/seq_query_split/ --train_dir="./aem_tmp/" --logging_dir="./aem_log/" --subsampling_rate=0.0001 --window_size=3 --embed_size=300 --negative_sample=5 --batch_size=384 --learning_rate=0.5 --max_train_epoch=20 --L2_lambda=0.005 --steps_per_checkpoint=400 --setting_file="./tests/example/AEM_exp.yaml"
