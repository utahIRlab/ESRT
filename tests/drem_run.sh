#train the model
cd ..
#python -m tests.main_v2 --data_dir=amazon_cellphone_index_dataset/cell_phone/ --input_train_dir=amazon_cellphone_index_dataset/cell_phone/random_query_split/ --logging_dir="./drem_log/" --train_dir="./drem_tmp/" --subsampling_rate=0.0001 --window_size=3 --embed_size=300 --negative_sample=5 --learning_rate=0.5 --batch_size=384 --max_train_epoch=20 --L2_lambda=0.005 --steps_per_checkpoint=400 --setting_file="./tests/example/DREM_exp.yaml"

#change the name of log file
#mv "./log/" "./log_$( date +%H:%M:%d-%m)/”

python main.py --setting_file="./tests/example/DREM_exp.yaml"
