#./galago-3.16-bin/bin/galago eval --judgments= ../amazon_cellphone_index_dataset/cell_phone/random_query_split/test.qrels --runs+ ../hem_tmp/test.bias_product.ranklist  --metrics+recip_rank --metrics+ndcg10 --metrics+P10
cd ..
../galago-3.16-bin/bin/galago eval --judgments= ../amazon_cellphone_index_dataset/seq_min_count5/seq_query_split/test.qrels --runs+ ./aem_tmp/test.bias_product.ranklist  --metrics+recip_rank --metrics+ndcg10 --metrics+P10


