# Overview
This is a collecetion of implements of following models, but it still under development:
* [HEM] Qingyao Ai, Yongfeng Zhang, Keping Bi, Xu Chen, W. Bruce Croft. 2017. Learning a Hierarchical Embedding Model for Personalized ProductSearch. In Proceedings of SIGIR ’17
* [AEM] Qingyao Ai, Daniel Hill, Vishy Vishwanathan and W. Bruce Croft. A Zero Attention Model for Personalized Product Search. Accepted in Proceedings of the 28th ACM International Conference on Information and Knowledge Management (CIKM’19) 
* [DREM] Qingyao Ai, Yongfeng Zhang, Keping Bi, W. Bruce Croft. Explainable Product Search with a Dynamic Relation Embedding Model. ACM Transactions on Information Systems (TOIS). 2019

These models are deep neural network models that jointly learn latent representations for queries, products and users and knowledge entites(DREM). 
They are designed as generative models and the embedding representations for queries, users and items in the models are learned through optimizing the log likelihood of observed user-query-item purchases. 
The probability (which is also the rank score) of an item being purchased by a user with a query can be computed with their corresponding latent representations. 
Please refer to the paper for more details.
### Requirements: ###
    1. To run the models, python 2.7+ and Tensorflow v1.0+ are needed. (In the paper, we used python 2.7.12 and Tensorflow v2.0.0)
    2. To run the jar package in ./seq_utils/AmazonDataset/jar/, JDK 1.7 is needed

### Run Models ###
**Data preparation**
```
cd experiment/
bash data_preprocess.sh
```
**Train model**
``` 
bash hem_run.sh # run hem model
# or
bash aem_run.sh # run aem model
# or
bash drem_run.sh # run drem model
```
**Test model**
``` 
bash hem_test.sh # run hem model
# or
bash aem_test.sh # run aem model
# or
bash drem_test.sh # run drem model
```
**Evaluate model**
``` 
bash hem_metric.sh # run hem model
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

### Result
|Models |MRR |NCDG10|P10|
|---:|:---:|:---:|:---:|
| HEM | 0.073|0.083|0.016| 
| AEM | 0.081|0.091|0.018|
| DREM| 0.101|0.114| 0.021|  
