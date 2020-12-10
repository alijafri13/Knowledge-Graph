#!/usr/bin/env python
# coding: utf-8

import json
import numpy as np
import spacy
from spacy import displacy
import ast
nlp = spacy.load("en_core_web_sm")

# Neuralcoref
import neuralcoref
nlp_coref = spacy.load('en')
neuralcoref.add_to_pipe(nlp_coref)

# Flair
# from flair.data import Sentence
# from flair.models import MultiTagger
# from flair.tokenization import SciSpacyTokenizer
# from flair.tokenization import SciSpacySentenceSplitter

# Stanford Open IE
from openie import StanfordOpenIE
from stanfordnlp.server import CoreNLPClient

import re

# Triple Filtering
import nltk
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

from word2number import w2n

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))

# Knowledge Graph Creation
import networkx as nx
import pygraphviz
G = nx.DiGraph()


# ### **Convert a file to a list of strings, one for each line in the file.**


import ast
def pull_data_from_file(file_to_read):
    '''
    INPUT: file_to_read (.txt, .json, .csv ... anything under the sun)
    OUTPUT: list(str)
    '''
    file=open(file_to_read, "r")
    contents = file.read()
    dictionary = ast.literal_eval(contents)
    text = dictionary['text']
    return text

#     lst = []
#     file = open(file_to_read, "r")
#     for line in file:
#         lst.append(line)
#     return lst

# Lowercases all words and ensures proper spacing after punctuation marks for a given text file.

def lower_space(full_txt):
    '''
    INPUT: list(str)
    OUTPUT: cleaned string
    '''
#    full_txt = ' '.join(full_txt)
    full_txt = full_txt.lower()
    return afterspace(full_txt)

# Ensures a space follows each punctuation mark.
def afterspace(text):
    return re.sub(r'(?<=[.,?!])(?=[^\s])', r' ', text)

# Ensures there is no space prior to each punctuation mark.
def beforespace(text):
    return re.sub(r'\s([?.!")%](?:\s|$))', r'\1', text)


# ### **Add Coreference Resolution to Text.**
#
# Follow readme details: https://github.com/huggingface/neuralcoref
#
# If warning pops up and kernel dies while running coref, paste following 2 blocks of code into terminal one at a time.
#
# <br>!git clone https://github.com/huggingface/neuralcoref.git
# <br>!pip install -U spacy
# <br>!python -m spacy download en
#
# <br>%cd neuralcoref
# <br>!pip install -r requirements.txt
# <br>!pip install -e


def neural_coref(clean_txt):
    '''
    INPUT: cleaned string
    OUTPUT: cleaned string with pronoun resolution
    '''
    doc = nlp_coref(clean_txt)
    no_pronoun_txt = doc._.coref_resolved
    return no_pronoun_txt


# ### **Runs Flair.**

# sentence = Sentence(new_clean_txt, use_tokenizer=SciSpacyTokenizer())
# tagger = MultiTagger.load("hunflair")
# tagger.predict(sentence)

# for entity in sentence.get_spans():

# splitter = SciSpacySentenceSplitter()
# sentences_split = splitter.split(clean_txt)
# tagger.predict(sentences_split)


# for line in sentences_split:
#     print(line.to_tagged_string())


# ### **Run Stanford Open IE.**
#
# Stanford Open IE website (in Java): https://nlp.stanford.edu/software/openie.html#Usage
#
# Python Wrapper for Stanford Open IE: https://github.com/philipperemy/Stanford-OpenIE-Python

# In[3]:


def launch_open_IE():
    return StanfordOpenIE()


# Runs Stanford Open IE
def stanford_open_IE(text, model):
    '''
    INPUT: cleaned list(str)
    OUTPUT: list of dictionary triples of 'subject', 'relation', and 'object'
    '''
    triples_dict = []
    for triple in model.annotate(text):
        triples_dict.append(triple)
    return triples_dict


# ## **Triple Filtering.**
#
# Steps to take:
# 1. Remove triples where:
#     <br> a. 'Subject' is a pronoun (POS tags: PRP, PRP$)
#     <br> b. 'Subject' is a verb in gerund form such as involving, taking, etc. (POS tags: VBG)
# 2. From 'subject', 'relation', and 'object' phrases, remove a) and b). If there is nothing left in the phrase after word removals, remove entire triple.
#     <br> a. Stop words
#     <br> b. Number values (works for numeric and word form)
# 3. Lemmatize the 'relation' phrase.
#
# List of Part-of-Speech tag abbreviations: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
# <br> (With examples): https://medium.com/@gianpaul.r/tokenization-and-parts-of-speech-pos-tagging-in-pythons-nltk-library-2d30f70af13b

# In[10]:


# Helper functions for Triple Filtering

# Function for Steps 1a and 1b
def subj_tester(subj_tokens):
    '''
    INPUT: 'subject' phrase in tokenized form
    OUTPUT: returns 0 or 1 value depending on if failed or passed filter
    '''
    subjwithTags = nltk.pos_tag(subj_tokens)
    for double in subjwithTags:
        if double[1] == 'PRP' or double[1] == 'PRP$' or double[1] == 'VBG':
            return 0
    return 1


# Function for Steps 2a and 2b for 'subject' and 'object' phrases
def stop_remover(subj_obj_tokens):
    '''
    INPUT: 'subject' and 'object' phrases in tokenized form
    OUTPUT: returns 0 value if failed filter, otherwise returns original phrase without any stop words
    '''
    noStop_phrase = ''
    for w in subj_obj_tokens:
        if w not in stop_words:
            try:
                numberChecker = w2n.word_to_num(w)
                continue
            except:
                noStop_phrase += w + ' '
    if noStop_phrase == '':
        return 0

    no_space = noStop_phrase[:-1]
    return no_space


# Function for Steps 2a, 2b, and 3 for 'relation' phrase
def lem_stop_rel(rel_tokens):
    '''
    INPUT: 'relation' phrase in tokenized form
    OUTPUT: returns 0 value if failed filter, otherwise returns 'relation' phrase in lemmatized form
    and without any stop words
    '''
    relation_lems = ''
    for w in rel_tokens:
        lemmedWord = wordnet_lemmatizer.lemmatize(w, pos='v')
        if lemmedWord not in stop_words:
            try:
                numberChecker = w2n.word_to_num(lemmedWord)
                continue
            except:
                relation_lems += lemmedWord + ' '
    if relation_lems == '':
        return 0

    no_space = relation_lems[:-1]
    return no_space


# In[11]:


# Actual filtering function for entire dictionary:

def triple_filter(triples_dict):
    '''
    INPUT: list of dictionary triples of 'subject', 'relation', and 'object'
    OUTPUT: filtered list of lists of subject, relation, and object
    '''
    filtered_list = []

    for triple in triples_dict:
        sub, rel, obj = triple.values()

        subj_tokens = word_tokenize(sub)
        rel_tokens = word_tokenize(rel)
        obj_tokens = word_tokenize(obj)

        # Step 1a, 1b
        if subj_tester(subj_tokens) == 0:
            continue


        # Step 2 for 'subject' and 'object' phrases
        noStop_subj = stop_remover(subj_tokens)
        noStop_obj = stop_remover(obj_tokens)

        if noStop_subj == 0 or noStop_obj == 0:
            continue
        else:
            triple['subject'] = noStop_subj
            triple['object'] = noStop_obj


        # Steps 2, and 3 for 'relation' phrase
        noStop_rel = lem_stop_rel(rel_tokens)
        if noStop_rel == 0:
            continue
        else:
            triple['relation'] = noStop_rel


        filtered_list.append(list(triple.values()))


    return filtered_list


# ### **Filtering of redundant triples.**

def remove_redundant(lst):
    '''
    INPUT: filtered list of lists of subject, relation, and object
    OUTPUT: same list with redudancies removed
    '''
    subjects = set()
    relationship = set()
    objects = set()
    redundant_removed = []
    for triple in lst:
        if len(triple) != 3:
            continue;
        elif triple[0] in subjects and triple[1] in relationship and triple[2] in objects:
            continue;
        elif "%" in triple or "\\" in triple:
            continue;
        else:
            redundant_removed.append(triple)
            subjects.add(triple[0])
            relationship.add(triple[1])
            objects.add(triple[2])
    return redundant_removed


# ### **Creates a .png and .dot file of the knowledge graph given by the triples list .**
# The files can be found in the directory this notebook is in.

# In[1]:


lst = [['subject', 'relationship', 'object'], ['i', 'like', 'pizza', None]]
type(lst) == list

None in lst[1]


def KG_creator(filtered_list):
    '''
    INPUT: filtered list of lists of subject, relation, and object
    OUTPUT: none
    '''
    for lst in filtered_list:
        #print(lst)

        if type(lst) == list and len(lst) == 3 and ("%" or "//" not in lst):
            subj_ = str(lst[0]).replace("%", "")
            obj_ = str(lst[2]).replace("%", "")
            G.add_edge(" "+subj_+" ", " "+obj_+" ", label=(" "+str(lst[1])+" "))
            print(lst)

    A_ = nx.nx_agraph.to_agraph(G)
    A_.layout()
    A_.draw('final_sample1' + '.png')
    A_.write('final_sample1' + '.dot')




#KG_creator(lst)
