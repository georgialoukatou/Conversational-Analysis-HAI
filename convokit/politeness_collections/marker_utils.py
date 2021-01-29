import json 
import pkg_resources
import os
from collections import defaultdict

# TODO： consider to raise error when path not provided 
UNIGRAM_FILE = pkg_resources.resource_filename("convokit",
    os.path.join("politeness_local/lexicons", "unigram_strategies.json"))

NGRAM_FILE = pkg_resources.resource_filename("convokit",
    os.path.join("politeness_local/lexicons", "ngram_strategies.json"))


START_FILE = pkg_resources.resource_filename("convokit",
    os.path.join("politeness_local/lexicons", "start_strategies.json"))



# Loading basic markers 
def load_basic_markers(unigram_path=None, ngram_path=None, start_path=None):
    
    # load unigram markers 
    if unigram_path is None:
        unigram_path = UNIGRAM_FILE
    
    if ngram_path is None:
        ngram_path = NGRAM_FILE
    
    if start_path is None:
        start_path = START_FILE
    
    with open(unigram_path, "r") as f:
        unigram_dict = json.load(f)

    with open(ngram_path, "r") as f:
        ngram_dict = json.load(f)

    # load phrase markers 
    with open(start_path, "r") as f:
        start_dict = json.load(f)
    
    return unigram_dict, start_dict, ngram_dict



############# Helper functions and variables #################

def extract_unigram_markers(sent_parsed, sent_idx, unigram_markers):
    
    return [(info['tok'], sent_idx, idx) for idx, info in enumerate(sent_parsed) if info['tok'].lower() in unigram_markers]


def extract_ngram_markers(sent_parsed, sent_idx, ngram_markers):
    
    ngrams_used = []
    words = [info['tok'].lower() for info in sent_parsed]
        
    for i, info in enumerate(sent_parsed[0:-1]):
        
        for ngram in ngram_markers:
            
            ngram_words = ngram.split()

            if words[i:i+len(ngram_words)] == ngram_words:
                
                start_idx = i
                ngrams_used.extend([(tok, sent_idx, start_idx+k) for k, tok in enumerate(ngram_words)])
                
    return ngrams_used


def extract_starter_markers(sent_parsed, sent_idx, starter_markers):
    
    start_tok = sent_parsed[0]['tok'].lower()
    
    if start_tok in starter_markers:
        return [(start_tok, sent_idx, 0)]
    else:
        return []

    

def extract_markers_from_sent(sent_parsed, sent_idx,\
                              unigram_markers, start_markers, ngram_markers,
                              marker_fns = None):
    '''
    Extracting markers from a parsed sentence 
    
    :param sent_parsed: parsed sentence 
    :param sent_idx: idx of the current sentence in the utterance 
    :param unigram_markers: set of unigram lexicon-based markers
    :param start_markers: set of lexicon-based markers at beginning of sentences 
    :param ngram_markers: set of ngram lexicon-based markers
    :param marker_fns: list of special functions to extract strategies not covered by lexicons 
    
    '''
    
    
    sent_summary = defaultdict(list)
    
    if marker_fns:
        marker_fns_names = [fn.__name__ for fn in marker_fns]
    else:
        marker_fns_names = None
    
    # unigram
    for k, unigrams in unigram_markers.items():
        
        if marker_fns_names is None or k not in marker_fns_names:
            sent_summary[k].extend(extract_unigram_markers(sent_parsed, sent_idx, unigrams))

    # ngrams
    for k, ngrams in ngram_markers.items():
        
        if marker_fns_names is None or k not in marker_fns_names:
            sent_summary[k].extend(extract_ngram_markers(sent_parsed, sent_idx, ngrams))
    
    # starter
    for k in start_markers:
        if marker_fns_names is None or k not in marker_fns_names:
            sent_summary[k].extend(extract_starter_markers(sent_parsed, sent_idx, start_markers[k]))
    
    
    # strategy by functions
    if marker_fns:
        for fn in marker_fns:
            sent_summary.update({fn.__name__: fn(sent_parsed, sent_idx)})

    
    return sent_summary