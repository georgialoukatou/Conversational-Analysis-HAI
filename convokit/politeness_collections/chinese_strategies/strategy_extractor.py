import json
import os
import re
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


PRAISE_PATTERN = re.compile(r'([真好]?\s?(厉害|棒|强|牛|美|漂亮))\s|(干\s?[得的]\s?真?\s?(好|漂亮))\s')
PLEASE_PATTERN = re.compile(r'([烦劳还]?\s?请)|([烦劳]\s?您)')
CAN_YOU_PATTERN = re.compile(r'[你您]\s?[是可想觉要].+?[吗呢呀？]')
COULD_YOU_PATTERN = re.compile(r'[你您]\s?(?P<A>[可想觉要])不(?P=A)')


PATTERNS = {"praise": PRAISE_PATTERN, "please": PLEASE_PATTERN, \
            "can_you": CAN_YOU_PATTERN, "could_you": COULD_YOU_PATTERN}


############# Strategies that require regex matching #################

def find_regex_strategies(pattern, tokens, sent_idx):
    
    # find matches 
    sent = " ".join(tokens)
    matches = [match.span() for match in re.finditer(pattern, sent)]
            
    marker_pos = []
    for match_start, match_end in matches:
        tok_start = len(sent[0:match_start].split())
        tok_end = len(sent[0:match_end].split()) 
        marker_pos.extend([(tok, tok_start+i, sent_idx) for i, tok in enumerate(tokens[tok_start:tok_end])])

    return marker_pos
    

def get_chinese_politeness_strategy_features(utt, unigram_markers = UNIGRAM_MARKERS,  \
                                             start_markers = START_MARKERS, \
                                             ngram_markers = NGRAM_MARKERS,
                                             patterns = PATTERNS):
    
    markers = defaultdict(list)
    parsed = [x["toks"] for x in utt.meta["parsed"]]
    
    
    for sent_idx, sent_parsed in enumerate(parsed):

        sent_markers = extract_markers_from_sent(sent_parsed, sent_idx, \
                                                 unigram_markers, start_markers, ngram_markers)
        
        # regex strategies
        tokens = [tok['tok'] for tok in sent_parsed]
        for name, pattern in patterns.items():
            sent_markers.update({name: find_regex_strategies(pattern, tokens, sent_idx)})
        
        # update markers 
        markers = {k:list(chain(markers[k], v)) for k,v in sent_markers.items()}
        
        
    features = {k: int(len(marker_list) > 0) for k, marker_list in markers.items()}
    
    return features, markers