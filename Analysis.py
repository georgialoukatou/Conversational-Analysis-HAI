
#!/usr/bin/env python
# -*- coding: utf-8 -*-



!pip install git+https://github.com/georgialoukatou/Cornell-Conversational-Analysis-Toolkit.git/@Hai  
#my github for extra pos and lemma analysis

import convokit
from convokit import Corpus, download

corpus = Corpus(download('subreddit-Cornell')) 
!python -m spacy download en_core_web_sm


##################

from convokit.text_processing import TextParser  
textparser = TextParser()
textparser.transform(corpus)     #perform SpaCy analysis

##################

#check use of PoS
import collections
convo = corpus.random_conversation() #pick random conversation from corpus 
d = collections.defaultdict(list)

d2 = collections.defaultdict(list)
for spkr in convo.iter_speakers():
   spkr_utts = list(spkr.iter_utterances())
   for i in spkr_utts:
      d[spkr.id].append(len(i.text.split(" "))) 
      for j in i.meta["parsed"]: 
        d2[spkr.id].append(list())
        for k in range(len(j["toks"])) :
          d2[spkr.id][-1].append(j["toks"][k]["pos"]) #find "parsed" pos metadata for each speaker (separate by utt)


for i in range(len(d2[spkr.id])):
    d2[spkr.id][i] = collections.Counter(d2[spkr.id][i])


#d2 gives parts of speech for each utterance and each speaker)

print(d2)

####################

#check length of utterances
import pandas as pd

df=pd.DataFrame.from_dict(d,orient='index').T #length of utterance for each speaker

print(df)

####################

# code minimally adapted from Doyle et al, 2016 / Yurovski et al., 2016

import operator, itertools
def group(utterances):
	utterances.sort(key=operator.itemgetter('convId'))
	list1 = []
	for key, items in itertools.groupby(utterances, operator.itemgetter('convId')):
		list1.append(list(items))
  
	return list1


def allMarkers(markers):
	categories = []
	for marker in markers:
		categories.append(marker)
	return categories

def checkMarkers(markers):
	toReturn = []
	for marker in markers:
		if isinstance(marker, str):
			toReturn.append({"marker": marker, "category": marker})
		else:
			toReturn.append(marker)
	return toReturn


def findMarkersInConvo(markers,convo):
	ba = {} # Number of times Person A and person B says the marker["marker"]
	bna = {}
	nbna = {}
	nba = {}
	for utterance in convo:				
		for j, marker in enumerate(markers):
			word = marker["marker"]
			msgMarker = word in utterance["msgMarkers"]
			replyMarker = word in utterance["replyMarkers"]
			
			if msgMarker and replyMarker:
				ba[word] = ba.get(word,0) + 1
			elif replyMarker and not msgMarker:
				bna[word] = bna.get(word,0) + 1
			elif not replyMarker and msgMarker:
				nba[word] = nba.get(word,0) + 1
			else:
				nbna[word] = nbna.get(word,0) + 1
	#		print(msgMarker, replyMarker)
			print(ba, bna, nba, nbna)
	return({'ba': ba,'bna': bna,'nba': nba,'nbna': nbna}) 
 
def metaDataExtractor(groupedUtterances, markers,corpusType=''):
	results = []
	for i, convo in enumerate(groupedUtterances):
				
		toAppend = findMarkersInConvo(markers,convo)		
		results.append(toAppend)
	return results

def readMarkers(markersFile,dialect=None):
	if dialect is None:
		reader = csv.reader(open(markersFile))
	else:
		reader = csv.reader(open(markersFile),dialect=dialect)
	markers = []
	print('marker\tcategory')
	for i, row in enumerate(reader):
		toAppend = {}
		toAppend["marker"] = row[0]
		if(len(row) > 1):
			toAppend["category"] = row[1]
		else:
			toAppend["category"] = row[0]
		markers.append(toAppend)
		#print(toAppend["marker"]+'\t'+toAppend["category"])
	return markers

def writeFile(results, outputFile, shouldWriteHeader):
	if len(results) == 0:
		print("x")
		return
	toWrite = []
	header = sorted(list(results[0].keys()))
	for row in results:
		toAppend = []
		for key in header:
			toAppend.append(row[key])
		toWrite.append(toAppend)
	if shouldWriteHeader:
		with open(outputFile, "w", newline='') as f:
			writer = csv.writer(f)
			writer.writerows([header])
		f.close()
	with open(outputFile, "a", newline='') as f:
		writer = csv.writer(f)
		writer.writerows(toWrite)
	f.close()

def determineCategories(msgMarkers,catdict,useREs=False):
	msgCats = []
	#iterate over catdict items {category: [words/REs]}
	for cd in catdict.items():
		if useREs:
			if any(any(wordre.match(marker) for marker in msgMarkers) for wordre in cd[1]):	#if REs, see if any tokens match each RE
				msgCats.append(cd[0])
		else:
			if any(word in msgMarkers for word in cd[1]):			#if just words, see if any word in category also in msg
				msgCats.append(cd[0])
	return msgCats

def makeCatDict(markers,useREs=False):
	mdict = {}
	for m in markers:
		marker = re.compile(''.join([m["marker"], '$'])) if useREs else m["marker"]
		if m["category"] in mdict:
			mdict[m["category"]].append(marker)
		else:
			mdict[m["category"]] = [marker]
		#mdict[m["category"]] = mdict.get(m["category"],[]).append(m["marker"])	#Need to swap marker and category labels
		#mdict[m["marker"]] = mdict.get(m["marker"],[]).append(m["category"])
	return(mdict)
 
def createAlignmentDict(category,result,smoothing,corpusType=''):
	toAppend = {}
	print(category)
	print("R", result)
	ba = int(result["ba"].get(category, 0))
	bna = int(result["bna"].get(category, 0))
	nbna = int(result["nbna"].get(category, 0))
	nba = int(result["nba"].get(category, 0))
	print(ba,bna,nbna,nba)
	#Calculating alignment only makes sense if we've seen messages with and without the marker
	if (((ba+nba)==0 or (bna+nbna)==0)):
		return(None)
	
	toAppend["category"] = category
		
	#Calculating Echoes of Power alignment 
	powerNum = ba
	powerDenom = ba+nba
	baseNum = ba+bna
	baseDenom = ba+nba+bna+nbna

	if(powerDenom != 0 and baseDenom != 0):
		dnmalignment = powerNum/powerDenom - baseNum/baseDenom
		toAppend["dnmalignment"] = dnmalignment
	else:
		toAppend["dnmalignment"] = False
	
	powerNum = ba
	powerDenom = ba+nba
	baseDenom = bna+nbna
	baseNum = bna
	powerProb = math.log((powerNum+smoothing)/float(powerDenom+2*smoothing))
	baseProb = math.log((baseNum+smoothing)/float(baseDenom+2*smoothing))
	alignment = powerProb - baseProb
	toAppend["alignment"] = alignment
	
	toAppend["ba"] = ba
	toAppend["bna"] = bna
	toAppend["nba"] = nba
	toAppend["nbna"] = nbna
	#print("toapp:", toAppend)
	return(toAppend)
 
def runFormula(results, markers, smoothing,corpusType):
	toReturn = []
	categories = allMarkers(markers)
	#print("h", results) 
	#print("c", categories)
	for i, result in enumerate(results):
		for j, category in enumerate(categories):
			toAppend = createAlignmentDict(category["category"],result,smoothing,corpusType)
		#	print("here", toAppend)   
			if toAppend is not None:
				toReturn.append(toAppend)
	toReturn = sorted(toReturn, key=lambda k: (k["speakerId"],k["replierId"],k["category"]))
	return toReturn

	###########################

	import pprint

# pair utterances from convokit 
markers = ["Hufflepuff", "Cornell", "Rowling", "absolutely", "Magic", "Howgarts", "the", "of", "very", "to", "through", "and","each", "other", "'d", "more", "less", "front", "back", "please", "any", "due", "just", "a", "in", "that", "for", "on", "are", "you", "it", "was", "were", "I", "as", "with", "they", "be", "at", "too", "have", "does", "this", "from", "or", "had", "by", "but", "some", "also", "what", "can", "out", "all", "your", "my", "up", "even", "so", "yes", "when", "almost", "no", "must", "should", "will", "would", "not"]


utterances =[]
previous_utt = dict()
#convo = corpus.random_conversation()

convo_utts= list(convo.iter_utterances()) 


for i in convo.traverse('bfs'): #iterate through utterances in tree-like (breadth) structure, level by level
  d = {} #each pair of speakers is a 'd' dictionary
  previous_utt[i.id] = (i.text,i.speaker.id)
  if i.reply_to == None:  #ignore if no reply
    continue
  d['convId'] = (previous_utt[i.reply_to][1], i.speaker.id)
  d['speakerId'] = previous_utt[i.reply_to][1] #?
  d['msg'] = previous_utt[i.reply_to][0]
  d['reply'] = i.text
  d['msgMarkers'] = []
  d['replyMarkers'] = []
  d['msgTokens']= []
  d['replyTokens']= []
  d["replierId"] = i.speaker.id
 # pprint.pprint(d)
  utterances.append(d)


for idx, utt in enumerate(utterances):  #append marker in utterance metadata if it exists
  for marker in markers:
			if marker in utt['msg']:
				utterances[idx]['msgMarkers'].append(marker)
			if marker in utt['reply']:
				utterances[idx]['replyMarkers'].append(marker)

#############################

pprint.pprint(utterances)

#############################

#main function - run alignment

smoothing=1
shouldWriteHeader=True
outputFile='aaaaaaaaa'
markers = checkMarkers(markers)
import math, csv
groupedUtterances = group(utterances)
#print(groupedUtterances)
metaData = metaDataExtractor(groupedUtterances,markers) 
#print(metaData)        
results = runFormula(metaData, markers, smoothing,'')
writeFile(results, outputFile, shouldWriteHeader)


##############################

