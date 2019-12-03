#test the model
cd ..
#python -m tests.main_v2 --data_dir=./amazon_cellphone_index_dataset/cell_phone/ --input_train_dir=./amazon_cellphone_index_dataset/cell_phone/random_query_split/ --logging_dir="./log_drem/" --train_dir="./tmp_drem/" --subsampling_rate=0.0001 --window_size=3 --embed_size=300 --negative_sample=5 --learning_rate=0.5 --batch_size=64 --max_train_epoch=20 --L2_lambda=0.005 --steps_per_checkpoint=400 --decode=True --similarity_func=bias_product --setting_file="./tests/example/DREM_exp.yaml"

#change the tmp/ name
#mv "./tmp/" "./tmp_$(date "+%Y-%m-%d %T")/"

python -m tests.main_v2 --setting_file="./tests/example/DREM_exp.yaml" --decode=True
