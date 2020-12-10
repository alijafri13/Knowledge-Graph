# Network Construction and Graph Mining from Multiple-Myeloma Secondary Data

## The Problem

Cancer researchers that are trying to understand some of the background work done in regards to a specific condition need to sift through thousands of papers. In fact, one search for multiple myeloma returns more than 57,000 results on PubMed. 
It is extremely time-intensive to try to understand the most important high-level links between the type of clinical trial, treatment, and result. Our solution applies data to this mass of information to ease the burden on the researcher.

## The Solution

We have built an automated pipeline that can take in hundreds of papers in PDF format and return a visual representation of the most important trends found throughout the papers. We call this visual representation a *knowledge graph*: the nodes are subjects and objects found in the text of the papers, and the edges in between are the relationships found between each. When viewed in a open-source bioinformatics software called Cytoscape, a researcher can type in a specific term they are looking for, and see how it relates to various proteins, patients, treatments, etc. that were found in the rest of the papers.

## Running the Entire Pipeline

Ensure you have the following files in the same directory: ```main.sh```, ```Functions.py```, ```Relation_Extraction.py```, ```pip_libraries.txt```, ```All_Functions.ipynb```, ```Text_Scraper.py```. 
We have written a shell script to run our entire pipeline on 1-100 papers in an automated fashion. This is labeled ```main.sh```. In order to run the script, navigate to the directory which contains your pdfs in a folder labeled 'PDF'. Within this directory too should include all the scripts labeled above. In your terminal open this directory and run ```bash main.sh``` This will set off a chain of functions explained in further detail below. 

*As a warning, a virtual environment will be set up that will take some time to load all the necessary packages. Depending on what you already have installed this could take upwards of 10 minutes. 

First, Text_Scraper.py..[insert high level overview of the whole pipeline]


## Overview of Coding Functions

### Flair: 
Flair is a state-of-the-art Named Entity Recognition (NER) tagger for text data. Our project specifically uses HunFlair, which is used for tagging biomedical texts. It comes with models for genes/proteins, chemicals, diseases, species, and cell lines. HunFlair builds on pretrained domain-specific language models that were trained on roughly 3 million full texts and about 25 million abstracts from the biomedical domain. The labels tagged by Flair give extra context to the user using our knowledge graph to help understand the different connections between diseases, genes, species, etc.

## Troubleshooting Fixes

Commands for libraries/packages unable to be installed using !pip install:\
brew install graphviz

Special:\
Use nltk.download() on a cell to install nltk packages (stem, corpus, tokenize) manually.


Coref Resolution:\
If warning pops up and kernel dies while running coref, paste following 2 blocks of code into terminal one at a time.

!git clone https://github.com/huggingface/neuralcoref.git  
!pip install -U spacy\
!python -m spacy download en


%cd neuralcoref\
!pip install -r requirements.txt\
!pip install -e

Stanford Open IE
Edited the CoreNLP server properties at this path: text-mining/knowledgegraph/kgvirtualenv/lib/python3.8/site-packages/openie/openie.py to 
include max_char_length=500000, timeout=100000. This avoids the Stanford CoreNLP Server crashing.
