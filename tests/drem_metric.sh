cd ..
../galago-3.16-bin/bin/galago eval --judgments= ../amazon_cellphone_index_dataset/seq_min_count5/seq_query_split/test.qrels --runs+ ./drem_tmp/test.bias_product.ranklist  --metrics+recip_rank --metrics+ndcg10 --metrics+P10
#./galago-3.16-bin/bin/galago eval --judgments= ./amazon_cellphone_index_dataset/min_count5/query_split/test.qrels --runs+ ./tmp/test.bias_product.ranklist  --metrics+recip_rank --metrics+ndcg10 --metrics+P10

