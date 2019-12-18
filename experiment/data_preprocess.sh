cd ../
# Download Amazon review dataset "Cell_Phones_and_Accessories" 5-core.
#wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Cell_Phones_and_Accessories_5.json.gz

# Download the meta data from http://jmcauley.ucsd.edu/data/amazon/
#wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Cell_Phones_and_Accessories.json.gz

# Stem and remove stop words from the Amazon review datasets if needed. Here, we stem the field of “reviewText” and “summary” without stop words removal.
java -Xmx4g -jar ./seq_utils/AmazonDataset/jar/AmazonReviewData_preprocess.jar false ./reviews_Cell_Phones_and_Accessories_5.json.gz ./reviews_Cell_Phones_and_Accessories_5.processed.gz

# Index datasets
python ./seq_utils/AmazonDataset/index_and_filter_review_file.py reviews_Cell_Phones_and_Accessories_5.processed.gz ./amazon_cellphone_index_dataset/ 5

# Match the meta data with the indexed data to extract queries:
java -Xmx16G -jar ./seq_utils/AmazonDataset/jar/AmazonMetaData_matching.jar false ./meta_Cell_Phones_and_Accessories.json.gz ./amazon_cellphone_index_dataset/seq_min_count5/

# Gather knowledge from meta data:
python ./seq_utils/AmazonDataset/match_with_meta_knowledge.py ./amazon_cellphone_index_dataset/seq_min_count5/ meta_Cell_Phones_and_Accessories.json.gz

# Sequently split train/test
python ./seq_utils/AmazonDataset/sequentially_split_train_test_data.py ./amazon_cellphone_index_dataset/seq_min_count5/ 0.3 0.3


