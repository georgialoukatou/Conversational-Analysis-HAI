import json
import os
from itertools import chain 
import pkg_resources
from collections import defaultdict

from convokit.politeness_collections.marker_utils import * 

LEXICON_DIR = "politeness_collections/chinese_strategies/lexicons"

UNIGRAM_FILE = pkg_resources.resource_filename("convokit",
    os.path.join(LEXICON_DIR, "unigram_strategies.json"))

NGRAM_FILE = pkg_resources.resource_filename("convokit",
    os.path.join(LEXICON_DIR, "ngram_strategies.json"))

START_FILE = pkg_resources.resource_filename("convokit",
    os.path.join(LEXICON_DIR, "start_strategies.json"))


UNIGRAM_MARKERS, START_MARKERS, NGRAM_MARKERS = load_basic_markers(unigram_path=UNIGRAM_FILE, \
                                                                   ngram_path=NGRAM_FILE, \
                                                                   start_path=START_FILE)


def get_chinese_politeness_strategy_features(utt, unigram_markers = UNIGRAM_MARKERS,  \
                                                  start_markers = START_MARKERS, \
                                                  ngram_markers = NGRAM_MARKERS):
    
    markers = defaultdict(list)
    parsed = [x["toks"] for x in utt.meta["parsed"]]
    
    
    for sent_idx, sent_parsed in enumerate(parsed):

        sent_markers = extract_markers_from_sent(sent_parsed, sent_idx, \
                                                 unigram_markers, start_markers, ngram_markers)
        
        # update markers 
        markers = {k:list(chain(markers[k], v)) for k,v in sent_markers.items()}
        
        
    features = {k: int(len(marker_list) > 0) for k, marker_list in markers.items()}
    
    return features, markers