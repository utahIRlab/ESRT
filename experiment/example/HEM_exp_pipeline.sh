cd ../../

# Download Amazon review dataset "Cell_Phones_and_Accessories" 5-core.
#wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Cell_Phones_and_Accessories_5.json.gz

# Download the meta data from http://jmcauley.ucsd.edu/data/amazon/
#wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Cell_Phones_and_Accessories.json.gz

# Stem and remove stop words from the Amazon review datasets if needed. Here, we stem the field of “reviewText” and “summary” without stop words removal.
#java -Xmx4g -jar ./seq_utils/AmazonDataset/jar/AmazonReviewData_preprocess.jar false ./reviews_Cell_Phones_and_Accessories_5.json.gz ./reviews_Cell_Phones_and_Accessories_5.processed.gz

# Index datasets
#python ./seq_utils/AmazonDataset/index_and_filter_review_file.py reviews_Cell_Phones_and_Accessories_5.processed.gz ./seq_tmp_data/ 5

# Match the meta data with the indexed data to extract queries:
#java -Xmx16G -jar ./seq_utils/AmazonDataset/jar/AmazonMetaData_matching.jar false ./meta_Cell_Phones_and_Accessories.json.gz ./seq_tmp_data/min_count5/

# Gather knowledge from meta data:
#python ./seq_utils/AmazonDataset/match_with_meta_knowledge.py ./seq_tmp_data/min_count5/ meta_Cell_Phones_and_Accessories.json.gz

# Randomly split train/test
## The 30% purchases of each user are used as test data
## Also, we randomly sample 20% queries and make them unique in the test set.
#python ./seq_utils/AmazonDataset/random_split_train_test_data.py seq_tmp_data/min_count5/ 0.3 0.3
#python ./seq_utils/AmazonDataset/sequentially_split_train_test_data.py seq_tmp_data/min_count5/ 0.3 0.3

# Create output directory
#if ! [ -d "./tmp/" ]; then
#  mkdir tmp
#fi

# train the data
python -m tests.main_v2 --setting_file="./tests/example/HEM_exp.yaml"

# test the data
python -m tests.main_v2 --setting_file="./tests/example/HEM_exp.yaml" --decode=True

# Download and install galago
wget https://iweb.dl.sourceforge.net/project/lemur/lemur/galago-3.16/galago-3.16-bin.tar.gz
tar xvzf galago-3.16-bin.tar.gz

# Compute the ranking metric
./galago-3.16-bin/bin/galago eval --judgments= ./seq_tmp_data/min_count5/seq_query_split/test.qrels --runs+ ./hem_tmp/test.bias_product.ranklist  --metrics+recip_rank --metrics+ndcg10 --metrics+P10

