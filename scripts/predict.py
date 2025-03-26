"""
Script file used for performing inference with an existing model.
"""

import torch
import json
import nltk
from nltk.tokenize import sent_tokenize
import huggingface_hub

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

BIN_REPO = 'dlsmallw/NLPinitiative-Binary-Classification'
ML_REPO = 'dlsmallw/NLPinitiative-Multilabel-Regression'

## Class used to encapsulate and handle the logic for inference
class InferenceHandler:
    def __init__(self, api_token):
        self.api_token = api_token
        self.bin_tokenizer, self.bin_model = self.init_model_and_tokenizer(BIN_REPO)
        self.ml_regr_tokenizer, self.ml_regr_model = self.init_model_and_tokenizer(ML_REPO)
        nltk.download('punkt_tab')

    def get_config(self, repo_id):
        config = None
        if repo_id and self.api_token:
            config = huggingface_hub.hf_hub_download(repo_id, filename='config.json', token=self.api_token)
        return config

    ## Initializes a model and tokenizer for use in inference using the models path
    def init_model_and_tokenizer(self, repo_id):
        config = self.get_config(repo_id)
        with open(config) as config_file:
            config_json = json.load(config_file)
        model_name = config_json['_name_or_path']

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(repo_id, token=self.api_token)

        model.eval()
        return tokenizer, model

    ## Handles logic used to encode the text for use in binary classification
    def encode_binary(self, text):
        bin_tokenized_input = self.bin_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        return bin_tokenized_input

    ## Handles logic used to encode the text for use in multilabel regression
    def encode_multilabel(self, text):
        ml_tokenized_input = self.ml_regr_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        return ml_tokenized_input

    ## Handles text encoding for both binary classification and multilabel regression
    def encode_input(self, text):
        bin_inputs = self.encode_binary(text)
        ml_inputs = self.encode_multilabel(text)
        return bin_inputs, ml_inputs
    
    ## Handles performing the full sentiment analysis (binary classification and multilabel regression)
    def classify_text(self, input):
        result = {
            'text_input': input,
            'results': []
        }

        sent_res_arr = []
        sentences = sent_tokenize(input)
        for sent in sentences:
            text_prediction, pred_class = self.discriminatory_inference(sent)

            sent_result = {
                'sentence': sent,
                'binary_classification': {
                    'classification': text_prediction,
                    'prediction_class': pred_class
                },
                'multilabel_regression': None
            }

            if pred_class == 1:
                ml_results = {
                    "Gender": None,
                    "Race": None,
                    "Sexuality": None,
                    "Disability": None,
                    "Religion": None,
                    "Unspecified": None
                }

                ml_infer_results = self.category_inference(sent)
                for idx, key in enumerate(ml_results.keys()):
                    ml_results[key] = min(max(ml_infer_results[idx], 0.0), 1.0)

                sent_result['multilabel_regression'] = ml_results
            sent_res_arr.append(sent_result)

        result['results'] = sent_res_arr
        return result
    
    ## Handles logic for checking the binary classfication of the text 
    def discriminatory_inference(self, text):
        bin_inputs = self.encode_binary(text)

        with torch.no_grad():
            bin_logits = self.bin_model(**bin_inputs).logits

        probs = torch.nn.functional.softmax(bin_logits, dim=-1)
        pred_class = torch.argmax(probs).item()
        bin_label_map = {0: "Non-Discriminatory", 1: "Discriminatory"}
        bin_text_pred = bin_label_map[pred_class]

        return bin_text_pred, pred_class
    
    ## Handles logic for assessing the categories of discrimination
    def category_inference(self, text):
        ml_inputs = self.encode_multilabel(text)

        with torch.no_grad():
            ml_outputs = self.ml_regr_model(**ml_inputs).logits
        
        ml_op_list = ml_outputs.squeeze().tolist()

        results = []
        for item in ml_op_list:
            results.append(max(0.0, item))

        return results