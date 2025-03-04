"""
Script file used for performing inference with an existing model.
"""

from pathlib import Path
import torch
import json

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


## Class used to encapsulate and handle the logic for inference
class InferenceHandler:
    def __init__(self, bin_model_path: Path, ml_regr_model_path: Path):
        self.bin_tokenizer, self.bin_model = self.init_model_and_tokenizer(bin_model_path)
        self.ml_regr_tokenizer, self.ml_regr_model = self.init_model_and_tokenizer(ml_regr_model_path)

    ## Initializes a model and tokenizer for use in inference using the models path
    def init_model_and_tokenizer(self, model_path: Path):
        with open(model_path / 'config.json') as config_file:
            config_json = json.load(config_file)
        model_name = config_json['_name_or_path']
        model_type = config_json['model_type']

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, model_type=model_type)
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
    def classify_text(self, text):
        res_obj = {
            'raw_text': text,
            'text_sentiment': None,
            'numerical_sentiment': None,
            'category_sentiments': {
                'Gender': None,
                'Race': None,
                'Sexuality': None,  
                'Disability': None,
                'Religion': None,  
                'Unspecified': None
            }
        }

        text_prediction, pred_class = self.discriminatory_inference(text)
        res_obj['text_sentiment'] = text_prediction
        res_obj['numerical_sentiment'] = pred_class

        if pred_class == 1:
            ml_infer_results = self.category_inference(text)

            for idx, key in enumerate(res_obj['category_sentiments'].keys()):
                res_obj['category_sentiments'][key] = ml_infer_results[idx]

        return res_obj
    
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
