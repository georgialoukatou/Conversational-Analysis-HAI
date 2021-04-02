#!/usr/bin/env python
# -*- coding: utf-8 -*-


from tqdm import tqdm
from convokit import Corpus, Speaker, Utterance
from collections import defaultdict

data_dir = "/Users/lscpuser/Documents/Conversational-Analysis-HAI/"

column_meta = {}

with open(data_dir + "pilotB_chat.csv", "r", encoding='utf-8', errors='ignore') as f:
	info = f.readlines()
	#print(info)
	for line in info:
  	 	column = [piece.strip() for piece in line.split(",")]
  	 	column_meta[column[0]] = {"gameId": column[4],
                                "target_num": column[6],
                                "repNum": column[7],
                                "trialNum": column[8],
                                "text": column[9],
                                "playerId": column[10],
                                "target": column[11],
                                "role": column[12]}
  	 	
print(column_meta)     
 
corpus_speakers = {k: Speaker(id = k, meta = v) for k,v in column_meta.items()}





# open(data_dir + "movie_characters_metadata.txt", "r", encoding='utf-8', errors='ignore') as f:
 #   speaker_data = f.readlines()
