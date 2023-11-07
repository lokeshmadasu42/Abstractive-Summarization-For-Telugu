## Abstractive Summarization For Telugu

This repository contains the code implementation of paper titled ["***TeSum: Human-Generated Abstractive Summarization Corpus for Telugu***"](https://aclanthology.org/2022.lrec-1.614.pdf) published in LREC-2022.

## Overview


## Data

Please convert the data into required format by running data_format.py.

## Models

We use modified fork of [huggingface transformers](https://github.com/huggingface/transformers) for fine-tuning the transformer models such as mT5, mBART and IndicBART.
Please check the following example script to fine-tune and evaluate the models. All the arguments are self-explanatory.  

bash 
```
python run_summarization.py \
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

bash
```
python3 run_adapter.py \
    --model_name_or_path facebook/mbart-large-50 \
    --do_train \
    --do_eval \
    --do_predict False\
    --train_adapter \
    --lang te_IN \
    --train_file train.csv \
    --validation_file dev.csv \
    --test_file test.csv \
    --max_source_length 512 \
    --max_target_length 256 \
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

bash
```
python3 run_adapter.py \
    --model_name_or_path facebook/mbart-large-50 \
    --do_train False\
    --do_eval False\
    --do_predict \
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
    --max_target_length 256 \
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

The following table contains ROUGE-L scores of various baseline models tested on TeSum test set. We use the forked version of [Multilingual ROUGE metric](https://github.com/csebuetnlp/xl-sum/tree/master/multilingual_rouge_scoring) for calculating the ROUGE scores.

| Model             | R-1   | R-2   | R-L   |
|-------------------|-------|-------|-------|
| Pointer Generator | 39.37 | 22.72 | 32.15 |
| mT5-small         | 37.42 | 20.82 | 30.88 |
| mBART             | 37.42 | 20.82 | 30.88 |
| mBART+Adapters    | 37.42 | 20.82 | 30.88 |
| IndicBART         | 37.42 | 20.82 | 30.88 |

***Hyper-parameters***

| PARAMETERS           | Pointer Generator | mT5              | mBART            | IndicBART        |
|----------------------|-------------------|------------------|------------------|------------------|
| Max source length    | 400               | 512              | 512              | 512              |
| Max target length    | 100               | 256              | 256              | 256              |
| Min target length    | 35                | 30               | 30               | 30               |
| Batch Size           | 8                 | 2                | 2                | 2                |
| Epoch/Iterations     | 100k iter         | 10 epochs        | 10 epochs        | 10 epochs        |
| Vocab Size           | 50k               | 250112           | 250112           | 250112           |
| Beam Size            | 4                 | 4                | 4                | 4                |
| Learning Rate        | 0.15              | 5.00E-04          | 5.00E-04          | 5.00E-04          |


