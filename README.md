### Run Models ###
**Data preparation**
```
cd experiment/
bash data_preprocess.sh
```
**Train model**
``` 
bash hem_run.sh # hem model
# or
bash aem_run.sh # run aem model
# or
bash drem_run.sh # run drem model
```
**Test model**
``` 
bash hem_test.sh # hem model
# or
bash aem_test.sh # run aem model
# or
bash drem_test.sh # run drem model
```
**Evaluate model**
``` 
bash hem_metric.sh # hem model
# or
bash aem_metric.sh # run aem model
# or
bash drem_metric.sh # run drem model
```

### Example Parameter Setting
|Hyper-parameters |HEM |AEM|DREM|
|---:|:---:|:---:|:---:|
| subsampling_rate | 0.0001|0.0001|0.0001|
| max_train_epoch | 20|20|20|
| rank_cutoff| 100|100| 100|
| window_size |5|5|5|
| embed_size |100|100|100
| max_gradient_norm|5.0|5.0|5.0|
| init_learning_rate|0.5|0.5|0.5|
| L2_lambda|0.005|0.005|0.005|
| query_weight |0.5|0.5|0.5| 
| negative_sampele |5|5|5|
| net_struct|"simplified_fs"|"simplified_fs"|"simplified_fs"|
| similarity_func|"bias_product"|"bias_product"|"bias_product"|
|batch_size|64|64|64|
|user_struct||"asin_attention"||
|num_heads||5||
|attention_func||'default'||
|max_history_length||10||

### Evaluation
|Models |MRR |NCDG10|P10|
|---:|:---:|:---:|:---:|
| HEM | 0.073|0.083|0.016| 
| AEM | 0.081|0.091|0.018|
| DREM| 0.101|0.114| 0.021|  
