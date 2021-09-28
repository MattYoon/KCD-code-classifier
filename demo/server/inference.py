import os
from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer
import torch
import argparse
import numpy as np
from pprint import pprint

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import sys


def load_model(args, model_name=None):
    if not model_name:
        model_name = args.model_name
    model_path = os.path.join(args.model_dir, model_name)
    print("Loading Model from:", model_path)
    load_state = torch.load(model_path)
    # load_state = torch.load(model_name)

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        args.config_name
        if args.config_name
        else args.model_name_or_path,
    )

    config.num_labels = 1084

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    ).to(args.device)

    model.load_state_dict(load_state['state_dict'], strict=True)

    print("Loading Model from:", model_path, "...Finished.")

    return model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--topk', default=1, type=int)
    parser.add_argument('--model_name_or_path', default = "/opt/ml/koelectra-korquad/output/checkpoint-7876", type=str, help='Path to pretrained model or model identifier from huggingface.co/models')
    parser.add_argument('--config_name', default = None, type=str, help='Pretrained config name or path if not the same as model_name')
    parser.add_argument('--tokenizer_name', default = None, type=str, help='Pretrained tokenizer name or path if not the same as model_name')

    args = parser.parse_args()

    return args


def inference_main():
    args = parse_args()
    args.model_name = "temp"
    preprocess = Preprocess(args)
    preprocess.load_test_data()
    test_data = preprocess.test_data

    print(f"size of test data : {len(test_data)}")
    torch.cuda.empty_cache()
    # del model
    inference(args, test_data)

inference_main()