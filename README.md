
# Gradient Cached Dense Passage Retrieval  
  
Gradient Cached Dense Passage Retrieval (`GC-DPR`) - is an extension of the original [DPR](https://github.com/facebookresearch/DPR) library.  
We introduce Gradient Cache technique which enables scaling batch size of the contrastive form loss and therefore the number of in-batch negatives far beyond GPU RAM limitation. With `GC-DPR` , you can reproduce the state-of-the-art open Q&A system trained on 8 x 32GB V100 GPUs with a single 11 GB GPU.  

To use Gradient Cache in your own project, checkout our [GradCache package](https://github.com/luyug/GradCache).

## Gradient Cache Technique  
- Retriever training quality depends on its effective batch size.
- The contrastive form loss (NLL with in-batch negatives) conditions on the entire batch and requires fitting the entire batch into GPU memory.
- Gradient Cache technique separates back propagation into loss to representation part and representation to encoder model part. It uses an extra forward pass without gradient tracking to compute representations and then store representations' gradients in cache. 
- The cached representation gradients remove data dependency in encoder back propagation and allows updating encoders one sub-batch at a time to fit GPU memory.

Details can be found in our paper [Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup
](https://arxiv.org/abs/2101.06983), to appeaer at The 6th Workshop on Representation Learning for NLP (RepL4NLP) 2021. Talk to us at ACL if you are interested!
 ```
@inproceedings{gao2021scaling,
      title={Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup},
      author={Luyu Gao, Yunyi Zhang, Jiawei Han, Jamie Callan},
      booktitle ={Proceedings of the 6th Workshop on Representation Learning for NLP},
      year={2021},
}
```
## Understanding the Code
One of the goals of this repo is to help clarify our gradient cache technique. The initial commit of the repo is a snapshot of `DPR`. You can run `diff` against it to see all changes made in `GC-DPR`. In general, we have modified train and inference codes. The model code is kept unchanged, and therefore users should be able to use trained checkpoints interchangeably between `DPR` and `GC-DPR`. 
 
## Additional Features
 Several changes were made to speed and/or facilitate easy reproduction of the original result. You may get some speed up using `GC-DPR` even when not using gradient.
 - Automatic Mix Precision (AMP): GC-DPR replaces apex AMP with Pytorch native AMP. It is supported for both training and inference.
 - Multi-process Data Loading:  batches are loaded using Pytorch's DataLoader during training and inference in loader processes, which largely avoids GPU idle waiting for text tokenization and batch creation.
 - Gradient Compensation: Pytorch's Distributed Data Parallel (DDP) average reduces model parameter gradients across GPUs while dense retriever training all-gather representations to compute loss. This means the actual gradient will be divided by the number of GPUs. For reproducibility, we scale/unscale gradient according to the number of available devices to replicate this effect. Another option is to use `_register_comm_hook` on the DDP object. Scaling loss is opted as it is easier to understand.


*The follwoing instructions are adapted from the orignial DPR's.*
## Installation  
You can clone this repo and install with pip.
```
git clone https://github.com/luyug/GC-DPR.git
cd GC-DPR
pip install .
```
GC-DPR is tested on Python 3.8.5, PyTorch 1.6.0 and Huggingface transformers 3.0.2.  
  
## Resources & Data formats  
You can follow the original DPR instructions:

First, you need to prepare data for either retriever or reader training. Each of the DPR components has its own input/output data formats. You can see format descriptions below. DPR provides NQ & Trivia preprocessed datasets (and model checkpoints) to be downloaded from the cloud using our data/download_data.py tool. One needs to specify the resource name to be downloaded. Run 'python data/download_data.py' to see all options.  
  
```bash  
python data/download_data.py \
   --resource {key from download_data.py's RESOURCES_MAP}  \
   [optional --output_dir {your location}]  
```  
The resource name matching is prefix-based. So if you need to download all data resources, just use --resource data.  
  
## Retriever input data format  
The data format of the Retriever training data is JSON.  
It contains pools of 2 types of negative passages per question, as well as positive passages and some additional information.  
  
```  
[  
  {  
   "question": "....",  
   "answers": ["...", "...", "..."],  
   "positive_ctxs": [{  
      "title": "...",  
      "text": "...."  
   }],  
   "negative_ctxs": ["..."],  
   "hard_negative_ctxs": ["..."]  
  },  
  ...  
]  
```  
  
Elements' structure  for negative_ctxs & hard_negative_ctxs is exactly the same as for positive_ctxs. The preprocessed data available for downloading also contains some extra attributes which may be useful for model modifications (like bm25 scores per passage). Still, they are not currently in use by DPR.  
  
You can download prepared NQ dataset used in the paper by using 'data.retriever.nq' key prefix. Only dev & train subsets are available in this format. We also provide question & answers only CSV data files for all train/dev/test splits. Those are used for the model evaluation since our NQ preprocessing step looses a part of original samples set. Use 'data.retriever.qas.*' resource keys to get respective sets for evaluation.  
  
```bash  
python data/download_data.py  
   --resource data.retriever  
   [optional --output_dir {your location}]  
```  
  
  
## Retriever training  
In order to start training on one machine:  
```bash  
python train_dense_encoder.py  
   --encoder_model_type {hf_bert | pytext_bert | fairseq_roberta}  
   --pretrained_model_cfg {bert-base-uncased| roberta-base}  
   --train_file {train files glob expression}  
   --dev_file {dev files glob expression}  
   --output_dir {dir to save checkpoints}  
   --grad_cache
   --q_chunk_size {question sub-batch sizes}
   --ctx_chunk_size {context sub-batch sizes}
   --fp16  # if train with mixed precision
```  
We introduce the following parameters to for gradient cached training,
- `--grad_cache` activates gradient cached training
- `--q_chunk_size` sub-batch size for updating the question encoder, default to 16
- `--ctx_chunk_size` sub-batch size for updating context encoder, default to 8

The default settings work for 11GB RTX 2080 ti.

## Retriever inference  
You can follow the original DPR instructions:
  
Generating representation vectors for the static documents dataset is a highly parallelizable process which can take up to a few days if computed on a single GPU. You might want to use multiple available GPU servers by running the script on each of them independently and specifying their own shards.  
  
```bash  
python generate_dense_embeddings.py \
   --model_file {path to biencoder checkpoint} \
   --ctx_file {path to psgs_w100.tsv file} \
   --shard_id {shard_num, 0-based} --num_shards {total number of shards} \
   --out_file ${out files location + name PREFX}  \
   --fp16  # if use mixed precision inference
```  
In addition, we provide the `--fp16` flag which will run encoder inference with mixed precision using the `autocast` context manager. Mixed precision inference gives roughly 2x speed up. The generated embeddings will be converted to full precision before saving for index search on x86 CPU.
  
## Retriever validation against the entire set of documents:  
  
```bash  
python dense_retriever.py \
   --model_file ${path to biencoder checkpoint} \
   --ctx_file  {path to all documents .tsv file} \
   --qa_file {path to test|dev .csv file} \
   --encoded_ctx_file "{encoded document files glob expression}" \
   --out_file {path to output json file with results} \
  --n-docs 200  
```  
We provide in addition flags `--encode_q_and_save`, `--q_encoding_path`, `--re_encode_q` which allows separation of question encoding and index search. One could therefore opt to run question encoding on a GPU server and search on a CPU server.
  
The tool writes retrieved results for subsequent reader model training into specified out_file.  
It is a json with the following format:  
  
```  
[  
    {  
        "question": "...",  
        "answers": ["...", "...", ... ],  
        "ctxs": [  
            {  
                "id": "...", # passage id from database tsv file  
                "title": "",  
                "text": "....",  
                "score": "...",  # retriever score  
                "has_answer": true|false  
     },  
]  
```  
Results are sorted by their similarity score, from most relevant to least relevant.  
  
## Optional reader model input data pre-processing.  

Since the reader model uses a specific combination of positive and negative passages for each question and also needs to know the answer span location in the bpe-tokenized form, it is recommended to preprocess and serialize the output from the retriever model before starting the reader training. This saves hours at train time.  
If you don't run this preprocessing, the Reader training pipeline checks if the input file(s) extension is .pkl and if not, preprocesses and caches results automatically in the same folder as the original files.  
  
```bash  
python preprocess_reader_data.py \
   --retriever_results {path to a file with results from dense_retriever.py} \
   --gold_passages {path to gold passages info} \
   --do_lower_case \
   --pretrained_model_cfg {pretrained_cfg} \
   --encoder_model_type {hf_bert | pytext_bert | fairseq_roberta} \
   --out_file {path to for output files} \
   --is_train_set  
```  
  
  
  
## Reader model training  
```bash  
python train_reader.py \
   --encoder_model_type {hf_bert | pytext_bert | fairseq_roberta} \
   --pretrained_model_cfg {bert-base-uncased| roberta-base} \
   --train_file "{globe expression for train files from #5 or #6 above}" \
   --dev_file "{globe expression for train files}" \
   --output_dir {path to output dir}  
```  
  
Notes:  
- if you use pytext_bert or fairseq_roberta, you need to download pre-trained weights and specify --pretrained_file parameter. Specify the dir location of the downloaded files for 'pretrained.fairseq.roberta-base' resource prefix for RoBERTa model or the file path for pytext BERT (resource name 'pretrained.pytext.bert-base.model').  
- Reader training pipeline does model validation every --eval_step batches  
- As the bi-encoder, it saves model checkpoints on every validation  
- Like the bi-encoder, there is no stop condition besides a specified amount of epochs to train.  
- Like the bi-encoder, there is no best checkpoint selection logic, so one needs to select that based on dev set validation performance which is logged in the train process output.  
- Our current code only calculates the Exact Match metric.  
  
## Reader model inference  
  
In order to make an inference, run `train_reader.py` without specifying `train_file`. Make sure to specify `model_file` with the path to the checkpoint, `passages_per_question_predict` with number of passages per question (being used when saving the prediction file), and `eval_top_docs` with a list of top passages threshold values from which to choose question's answer span (to be printed as logs). The example command line is as follows.  
  
```bash  
python train_reader.py \
  --prediction_results_file {some dir}/results.json \
  --eval_top_docs 10 20 40 50 80 100 \
  --dev_file {path to data.retriever_results.nq.single.test file} \
  --model_file {path to the reader checkpoint} \
  --dev_batch_size 80 \
  --passages_per_question_predict 100 \
  --sequence_length 350  
```  

  
## Best hyperparameter settings  
  
e2e example with the best settings for NQ dataset.  Most of this section follows DPR's instructions. The retriever training session is adjusted for a single 11 GB GPU.
  
### 1. Download all retriever training and validation data:  
  
```bash  
python data/download_data.py --resource data.wikipedia_split.psgs_w100  
python data/download_data.py --resource data.retriever.nq  
python data/download_data.py --resource data.retriever.qas.nq  
```  
  
### 2. Biencoder(Retriever) training in single set mode.  
  
Train on a single 11GB RTX 2080 ti GPU,
  
```bash  
python train_dense_encoder.py \
   --max_grad_norm 2.0 \
   --encoder_model_type hf_bert \
   --pretrained_model_cfg bert-base-uncased \
   --seed 12345 \
   --sequence_length 256 \
   --warmup_steps 1237 \
   --batch_size 128 \
   --do_lower_case \
   --train_file "{glob expression to train files downloaded as 'data.retriever.nq-train' resource}" \
   --dev_file "{glob expression to dev files downloaded as 'data.retriever.nq-dev' resource}" \
   --output_dir {your output dir} \
   --learning_rate 2e-05 \
   --num_train_epochs 40 \
   --dev_batch_size 16 \
   --val_av_rank_start_epoch 30 \
   --fp16 \
   --grad_cache \
   --global_loss_buf_sz 2097152 \
   --val_av_rank_max_qs 1000
```  
This takes less than 2 days to complete the training for 40 epochs. 
The best checkpoint for bi-encoder is usually the last, but it should not be so different if you take any after epoch ~ 25.  We use smaller value for `--val_av_rank_max_qs` to speed up validation.
  
### 3. Generate embeddings for Wikipedia.  
Just use instructions for "Generating representations for large documents set". It takes about 40 minutes to produce 21 mln passages representation vectors on 50 2 GPU servers.  
  
### 4. Evaluate retrieval accuracy and generate top passage results for each of the train/dev/test datasets.  
  
```bash  
python dense_retriever.py \
   --model_file {path to checkpoint file from step 1} \
   --ctx_file {path to psgs_w100.tsv file} \
   --qa_file {path to test/dev qas file} \
   --encoded_ctx_file "{glob expression for generated files from step 3}" \
   --out_file {path for output json files} \
   --n-docs 100 \
   --validation_workers 32 \
   --batch_size 64  
```  
  
Adjust batch_size based on the available number of GPUs, 64 should work for 2 GPU server.  
  
### 5. Reader training  
Here's the DPR's original instruction for reader training.
We trained reader model for large datasets using a single 8 GPU x 32 GB server.  
  
```bash  
python train_reader.py \
   --seed 42 \
   --learning_rate 1e-5 \
   --eval_step 2000 \
   --do_lower_case \
   --eval_top_docs 50 \
   --encoder_model_type hf_bert \
   --pretrained_model_cfg bert-base-uncased \
   --train_file "{glob expression for train output files from step 4}" \
   --dev_file {glob expression for dev output file from step 4} \
   --warmup_steps 0 \
   --sequence_length 350 \
   --batch_size 16 \
   --passages_per_question 24 \
   --num_train_epochs 100000 \
   --dev_batch_size 72 \
   --passages_per_question_predict 50 \
   --output_dir {your save dir path}  
```  
  
We found that using the learning rate above works best with static schedule, so one needs to stop training manually based on evaluation performance dynamics.  
Our best results were achieved on 16-18 training epochs or after ~60k model updates.  
  
We provide all input and intermediate results for e2e pipeline for NQ dataset and most of the similar resources for Trivia.  
  
## Misc.  
- TREC validation requires regexp based matching. We support only retriever validation in the regexp mode. See --match parameter option.  
- WebQ validation requires entity normalization, which is not included as of now.  
  
## Reference  
If you find `GC-DPR` helpful, please consider citing [our paper](https://arxiv.org/abs/2101.06983):
```
@inproceedings{gao2021scaling,
      title={Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup},
      author={Luyu Gao, Yunyi Zhang, Jiawei Han, Jamie Callan},
      booktitle ={Proceedings of the 6th Workshop on Representation Learning for NLP},
      year={2021},
}
```
Also consider citing the original [DPR paper](https://arxiv.org/abs/2004.04906):  
```  
@misc{karpukhin2020dense,  
    title={Dense Passage Retrieval for Open-Domain Question Answering},  
    author={Vladimir Karpukhin and Barlas Oğuz and Sewon Min and Patrick Lewis and Ledell Wu and Sergey Edunov and Danqi Chen and Wen-tau Yih},  
    year={2020},  
    eprint={2004.04906},  
    archivePrefix={arXiv},  
    primaryClass={cs.CL}  
}  
```  
  
## License  
GC-DPR inherits DPR's CC-BY-NC 4.0 licensed as of now.
