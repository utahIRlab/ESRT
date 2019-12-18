cd ..

# download galgo if note exist
if ! [ -d "./galago-3.16-bin/" ]; then
  wget https://iweb.dl.sourceforge.net/project/lemur/lemur/galago-3.16/galago-3.16-bin.tar.gz
  tar xvzf galago-3.16-bin.tar.gz
fi

# evaluate HEM
./galago-3.16-bin/bin/galago eval --judgments= "./amazon_cellphone_index_dataset/seq_min_count5/seq_query_split/test.qrels" --runs+ ./drem_tmp/test.bias_product.ranklist  --metrics+recip_rank --metrics+ndcg10 --metrics+P10

