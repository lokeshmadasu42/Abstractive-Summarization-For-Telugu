## Abstractive Summarization For Telugu

This repository contains the code implementation of paper titled ["***TeSum: Human-Generated Abstractive Summarization Corpus for Telugu***"](https://aclanthology.org/2022.lrec-1.614.pdf) published in LREC-2022.

## Overview

Abstractive summarization is the task of generating a condensed(short) version of an article by preserving the important information. We implemented state-of-the-art baseline models and tested their performance on TeSum data.

## Data

TeSum is the first ever, largest human annotated dataset for the Abstractive Summarization task in Telugu. The dataset contains 20,329 high-quality article-summary pairs. The train,dev and test splits are 16295, 2017 and 2017.

**Note**: Please convert the data into required format before passing it to models (run data_format.py).

## Models

We use the modified fork of [huggingface transformers](https://github.com/huggingface/transformers) for fine-tuning the transformer models such as mT5, mBART and IndicBART. Please check the following example script to fine-tune and evaluate the models. All the arguments are self-explanatory.  

```
python3 run_summarization.py \
    --model_name_or_path ai4bharat/IndicBART-XLSum \
    --do_train True \
    --do_eval True\
    --do_predict True\
    --lang te_IN \
    --train_file train.csv \
    --validation_file dev.csv \
    --test_file test.csv \
    --max_source_length 512 \
    --max_target_length 256 \
    --output_dir tmp/outputs/ \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --dataloader_num_workers 4 \
    --logging_strategy "epoch" \
    --save_strategy "no" \
    --overwrite_output_dir \
    --predict_with_generate \
    --num_train_epochs 10 \
    --forced_bos_token=tokenizer.lang_code_to_id["te_IN"] \
    --summary_column summary \
    --text_column text $@

```

***Fine-tune mBART+Adapters***

```
python3 run_adapter.py \
    --model_name_or_path facebook/mbart-large-50 \
    --do_train True\
    --do_eval True\
    --do_predict False\
    --train_adapter \
    --lang te_IN \
    --train_file train.csv \
    --validation_file dev.csv \
    --test_file test.csv \
    --max_source_length 512 \
    --max_target_length 200 \
    --learning_rate 1e-4 \
    --output_dir "tmp/outputs/" \
    --dataloader_num_workers 2 \
    --logging_strategy "epoch" \
    --save_strategy "no" \
    --fp16 True \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1\
    --overwrite_output_dir \
    --summary_column summary \
    --num_train_epochs 10 \
    --text_column text $@
```

***Evaluate mBART+Adapters***

```
python3 run_adapter.py \
    --model_name_or_path facebook/mbart-large-50 \
    --do_train False\
    --do_eval False\
    --do_predict True\
    --train_adapter \
    --load_adapter summarization \
    --adapter_config "pfeiffer" \
    --adapter_non_linearity "relu" \
    --adapter_reduction_factor 2 \
    --lang te_IN \
    --train_file train.csv \
    --validation_file dev.csv \
    --test_file test.csv \
    --max_source_length 512 \
    --max_target_length 200 \
    --learning_rate 1e-4 \
    --output_dir "tmp/outputs/" \
    --logging_strategy "epoch" \
    --save_strategy "no" \
    --fp16 True \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=2\
    --overwrite_output_dir \
    --summary_column summary \
    --num_train_epochs 1 \
    --predict_with_generate \
    --text_column text $@

```

***Pointer Generator***

We also trained Seq-Seq [pointer generator](https://github.com/atulkum/pointer_summarizer) model for our experiments. For further information on how to train and evaluate the model please go through the Pointer_Generator folder.

## Benchmarks

The following table contains ROUGE-L scores of various baseline models tested on TeSum test set. We use the forked version of [Multilingual ROUGE metric](https://github.com/csebuetnlp/xl-sum/tree/master/multilingual_rouge_scoring) for computing the ROUGE scores.

| Model             | R-1   | R-2   | R-L   |
|-------------------|-------|-------|-------|
| Pointer Generator | 39.37 | 22.72 | 32.15 |
| mT5-small         | 37.42 | 20.82 | 30.88 |
| mBART             | 41.10 | 24.10 | 33.76 |
| mBART+Adapters    | 41.41 | 24.00 | 34.00 |
| IndicBARTSS       | 39.15 | 21.83 | 32.10 |

***Hyper-parameters***:

| PARAMETERS           | Pointer Generator | mT5              | mBART            | IndicBARTSS        |
|----------------------|-------------------|------------------|------------------|------------------|
| Max source length    | 400               | 512              | 512              | 512              |
| Max target length    | 100               | 256              | 200              | 256              |
| Min target length    | 35                | 30               | 30               | 30               |
| Batch Size           | 8                 | 2                | 2                | 2                |
| Epoch/Iterations     | 100k iter         | 10 epochs        | 10 epochs        | 10 epochs        |
| Vocab Size           | 50k               | 250112           | 250054           | 64000            |
| Beam Size            | 4                 | 4                | 5                | 4                |
| Learning Rate        | 0.15              | 5.00E-04         | 1.00E-04         | 5.00E-04         |


