# Network Construction and Graph Mining from Multiple-Myeloma Secondary Data

## The Problem

Cancer researchers that are trying to understand some of the background work done in regards to a specific condition need to sift through thousands of papers. In fact, one search for multiple myeloma returns more than 57,000 results on PubMed. 
It is extremely time-intensive to try to understand the most important high-level links between the type of clinical trial, treatment, and result. Our solution applies data to this mass of information to ease the burden on the researcher.

## The Solution

We have built an automated pipeline that can take in hundreds of papers in PDF format and return a visual representation of the most important trends found throughout the papers. We call this visual representation a *knowledge graph*: the nodes are subjects and objects found in the text of the papers, and the edges in between are the relationships found between each. When viewed in a open-source bioinformatics software called Cytoscape, a researcher can type in a specific term they are looking for, and see how it relates to various proteins, patients, treatments, etc. that were found in the rest of the papers.

## Running the Entire Pipeline

Ensure you have the following files in the same directory: ```main.sh```, ```Functions.py```, ```Relation_Extraction.py```, ```pip_libraries.txt```, ```Functions.ipynb```, ```Text_Scraper.py```. 
We have written a shell script to run our entire pipeline on 1-100 papers in an automated fashion. This is labeled ```main.sh```. In order to run the script, navigate to the directory which contains your pdfs in a folder labeled 'PDF'. Within this directory too should include all the scripts labeled above. In your terminal open this directory and run ```bash main.sh``` This will set off a chain of functions explained in further detail below. 

*As a warning, a virtual environment will be set up that will take some time to load all the necessary packages. Depending on what you already have installed this could take upwards of 10 minutes. 

### 1. Extract XML Data from PDF (CERMINE)
First PDF data will be extracted in the form of an XML file using CERMINE github repository. Newly generated XML filed will be inputted into a new folder         in the working directory called XML. Conversion progress will be seen in your terminal
### 2. Clean XML data to text data (Text_Scraper.py)
XML data will then be filtered with a series of cleaners with BeautifulSoup that would remove any unecessary information besides text included in the             paper. Such removals would include header, footers, references, as well as unecessary tags in the way an XML file is written.
### 3. Relationship Extraction (Relation_Extraction.py)
Text data is then run through a series of primary steps to further orient the data for proper relationship extraction such as neural coreference                 resolution. Following this, taking advantage of Stanford NLP OpenIE relationship extraction software, subject, relationship, and object triples are               extracted from the paper as a whole. After extraction, to remove unecessary or simppligy noisy relationships, we utilize Python's NLTK library to                 lemmatize, remove stop words, as well as stem words to their root form. 
### 4. Knowledge Graph Creation
At the end of extraction, a visual representation of all this information is presented in a Knowledge Graph using Pygraphviz. Output files will be a .png
and a .dot file that can be loaded into Cytoscape an open source bioinformatics software platform for visualizing molecular interaction networks and
integrating with gene expression profiles and other state data. 



## Overview of Coding Libraries and Functions

### Neural Coreference Resolution:
NeuralCoref is an extension for spaCy's NLP pipeline, which annotates and resolves coreference clusters using a neural network. Given a pronoun incidence, NeuralCoref replaces the pronoun with its same entity, the main noun.\
Before Neural Coref: *Proteins* had a positive effect on some patients, but *they* harmed others.\
After Neural Coref: *Proteins* had a positive effect on some patients, but *proteins* harmed others.

### Stanford Open Information Extraction:
Open information extraction (open IE) refers to the extraction of relation tuples, typically binary relations, from plain text, such as (Mark Zuckerberg; founded; Facebook). The central difference from other information extraction is that the schema for these relations does not need to be specified in advance; typically the relation name is just the text linking two arguments. The system first splits each sentence into a set of entailed clauses. Each clause is then maximally shortened, producing a set of entailed shorter sentence fragments. These fragments are then segmented into OpenIE triples, and output as a {subject, relation, object} triple by the system.

### Triple Filtering using the Natural Language Toolkit Library:
Three main steps to remove redundant and unsuitable triples:
#### 1. Tokenization and Part-of-Speech Tagging (NLTK's POS tag)
After tokenizing a sentence word-by-word, each word is tagged with an abbreviated version of its respective part of speech. The function then removes the entire triple if it contains a pronoun (triples that made it past the Coref filter), or if the 'Subject' phrase contains a verb in gerund form (verbs that have -ing as a suffix, which are inappropriate nouns).
#### 2. Removal of All Stop Words (NLTK's stopwords corpus)
Stop words are extremely common words that hold no value/meaning such as 'the', 'and', or 'from'. By removing all stop words, we drastically reduce the number of words we need to process while still retaining most, if not all, of the original sentence's meaning.
#### 3. Lemmatization/Stemming of the 'Relation' Phrase (NLTK's wordnetlemmatizer)
Lemmatization of the 'Relation' phrase reverts all verbs back to their root form to group together verbs that are the same but have different tenses.


### Flair: 
Flair is a state-of-the-art Named Entity Recognition (NER) tagger for text data. Our project specifically uses HunFlair, which is used for tagging biomedical texts. It comes with models for genes/proteins, chemicals, diseases, species, and cell lines. HunFlair builds on pretrained domain-specific language models that were trained on roughly 3 million full texts and about 25 million abstracts from the biomedical domain. The labels tagged by Flair give extra context to the user using our knowledge graph to help understand the different connections between diseases, genes, species, etc.

