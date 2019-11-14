import numpy as np
from sklearn.model_selection import train_test_split
from convokit.model import Corpus, Conversation, User, Utterance
from sklearn import svm
from typing import List, Hashable, Callable, Union
from convokit import Transformer
import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import requests
import os
import sys
import random
import unicodedata
import itertools
from urllib.request import urlretrieve
from .craftUtil import *
from convokit import download, Corpus
from .craftModel import *

class CraftForecaster(Transformer):
    def __init__(self, device='cpu'):
        assert device in ['cuda', 'cpu']
        self.device = torch.device(device)
        self.voc = loadPrecomputedVoc("wikiconv", WORD2INDEX_URL, INDEX2WORD_URL)
        self.predictor = initialize_model(self.voc, self.device)
        self.threshold = 0.570617

    @staticmethod
    def evaluateBatch(encoder, context_encoder, predictor, voc, input_batch, dialog_lengths,
                      dialog_lengths_list, utt_lengths, batch_indices, dialog_indices, batch_size, device, max_length=MAX_LENGTH):
        # Set device options
        input_batch = input_batch.to(device)
        dialog_lengths = dialog_lengths.to(device)
        utt_lengths = utt_lengths.to(device)
        # Predict future attack using predictor
        scores = predictor(input_batch, dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices, dialog_indices, batch_size, max_length)
        predictions = (scores > 0.5).float()
        return predictions, scores

    @staticmethod
    def evaluateDataset(dataset, encoder, context_encoder, predictor, voc, batch_size, device):
        # create a batch iterator for the given data
        batch_iterator = batchIterator(voc, dataset, batch_size, shuffle=False)
        # find out how many iterations we will need to cover the whole dataset
        n_iters = len(dataset) // batch_size + int(len(dataset) % batch_size > 0)
        output_df = {
            "id": [],
            "prediction": [],
            "score": []
        }
        for iteration in range(1, n_iters+1):
            batch, batch_dialogs, _, true_batch_size = next(batch_iterator)
            # Extract fields from batch
            input_variable, dialog_lengths, utt_lengths, batch_indices, dialog_indices, convo_ids, target_variable, mask, max_target_len = batch
            dialog_lengths_list = [len(x) for x in batch_dialogs]
            # run the model
            predictions, scores = CraftForecaster.evaluateBatch(encoder, context_encoder, predictor, voc, input_variable,
                                                dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices, dialog_indices,
                                                true_batch_size, device)

            # format the output as a dataframe (which we can later re-join with the corpus)
            for i in range(true_batch_size):
                convo_id = convo_ids[i]
                pred = predictions[i].item()
                score = scores[i].item()
                output_df["id"].append(convo_id)
                output_df["prediction"].append(pred)
                output_df["score"].append(score)

            print("Iteration: {}; Percent complete: {:.1f}%".format(iteration, iteration / n_iters * 100))

        return pd.DataFrame(output_df).set_index("id")


