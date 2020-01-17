cd ..

# evaluate ZAM
./galago-3.16-bin/bin/galago eval --judgments= "./amazon_cellphone_index_dataset/seq_min_count5/seq_query_split/test.qrels" --runs+ ./zam_tmp/test.bias_product.ranklist  --metrics+recip_rank --metrics+ndcg10 --metrics+P10
