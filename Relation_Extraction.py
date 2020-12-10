#!/usr/bin/env python
# coding: utf-8

from Functions import *
import os
from tqdm import tqdm


#Iterate through all downloaded papers
directory = r'./TEXT/'
all_files = []
for file in os.scandir(directory):
    all_files.append(file.path)


## Launch Stanford Open IE
print('Initializing Stanford OpenIE...')
model = launch_open_IE()

first = True

for paper_name in tqdm(all_files):
    try:
        func1 = pull_data_from_file(paper_name)
    except UnicodeDecodeError:
        print(paper_name)

    full_txt = " ".join(func1)
    # print(full_txt)

    # print('\n'+ 'Applying coreference resolution...')
    #Apply coreference resolution
    # print(full_txt)

    # neural_coref = neural_coref(full_txt)

	#Extract relationships
    print('Extracting Relationships from' + paper_name)
    stanford_relation_extraction = stanford_open_IE(full_txt, model)
    
	#Filter relationships to remove redundancies
    triple_filtered = triple_filter(stanford_relation_extraction)
    cleaned_relationships = remove_redundant(triple_filtered)

    # Uncomment when using Flair
    # print('Adding Flair Labels...')
    # initialize_flair = flair_labels(neural_coref)
    # relationships_with_flair = adding_flair(initialize_flair,cleaned_relationships)


	#Add to list of aggregated relationships
    if first:
        aggregated_rels = cleaned_relationships
        first = False
        print(paper_name + ' completed')
    else:
        aggregated_rels.append(cleaned_relationships)
        print(paper_name + ' completed')

    # Use following lines when implementing Flair
    # if first:
    #     aggregated_rels = relationships_with_flair
    #     first = False
    #     print(paper_name + ' completed')
    # else:
    #     aggregated_rels.append(relationships_with_flair)
    #     print(paper_name + ' completed')

# print(aggregated_rels)
#Output final knowledge graph with aggregated relationships
KG_creator(aggregated_rels)
