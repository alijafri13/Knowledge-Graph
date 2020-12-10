#!/usr/bin/env python
# coding: utf-8

import os
import ntpath
from bs4 import BeautifulSoup
import regex as re
from PyPDF2 import PdfFileReader
import json
import _pickle as pickle


cwd = os.getcwd()

pdfs = []
for pdf in os.scandir(cwd + '/PDF'):
    pdfs.append(pdf.path)


list_of_words = ['author','authors','university',
                 'open access','journal','janssen','et al','et. al','amgen','.com','.org'
                ]



def get_files(file_format):
    files = []
    if file_format == '.pdf':
        directory = cwd + '/PDF'
    elif file_format =='.cermxml':
        directory = cwd + '/XML'
    elif file_format == '.txt':
        directory = cwd + '/Text'
    for filename in os.listdir(directory):
        if filename.endswith(file_format):
            files.append(os.path.join(directory, filename))
        else:
            continue
    return files

def length(string):
    length = len(string) - string.count(' ')
    return length

def remove_bad_words(bad_words, sentence):
    sent = sentence.split()
    for word in sent:
        if word in bad_words:
            return False
    return True


def lxml_to_text(xml_file):
    f = open(xml_file,'r')
    soup = BeautifulSoup(f.read(), "html.parser")
    for sec in soup.find_all('sec'):
        for p in sec.find_all('p'):
            text = str(p)
            head, tail = ntpath.split(xml_file)
            with open(os.path.join(cwd+'/Text',tail.replace('cermxml',"")+'txt'), "a") as file1:
                file1.write(text)


def remove_tags(text_file):
    file = open(text_file,'r+')
    lines = file.readlines()
    file.seek(0)
    long = [i for i in lines if length(i)>11]
    merge = "".join(long)
    text = merge.replace('\n'," ")
    text =text.replace('</p>',"\n\n")
    text = re.sub(r'<.+?>', '', text)
    text = re.sub(r"[\(\[].*?[\)\]]", "", text)
    clean = re.sub(r'\(.*?\)','',text)
    clean = re.sub(' +', ' ',clean)
    clean = clean.split("Acknowledgements")[0]
    file.writelines(clean)
    file.truncate()



def add_abstract(text_file):
    text = open(text_file,'r+')
    lines = text.readlines()
    text.seek(0)
    cermxml_file = cwd+'/XML/'+os.path.basename(text_file).replace('.txt','.cermxml')
    file = open(cermxml_file,'r')
    soup = BeautifulSoup(file.read(), "html.parser")
    abstract = (soup.find('abstract'))
    if abstract is None:
        abstract = ''
        file_1 = lines
        text.writelines(file_1)
        text.truncate()
    else:
        file_1 = [abstract.text+'\n\n'] + lines
        text.writelines(file_1)
        text.truncate()

def add_metadata(text_file):
    file = open(text_file, "r+")
    lines = file.readlines()
    file.seek(0)
    lines = [i for i in lines if len(i)>400]
    lines = [i.lower() for i in lines]
    lines = [i for i in lines if remove_bad_words(list_of_words,i)]
    text = ''.join(lines)
    text = text.replace("\n", "")
    text = text.replace('[',"")
    text = text.replace(']',"")
#     pdf_file_path = cwd+'/PDF/'+os.path.basename(text_file).replace('.txt','.pdf')
#     pdf_file_open = PdfFileReader(open(pdf_file_path,'rb'))
#     pdf_info = pdf_file_open.getDocumentInfo()
#     dictionary = { 'metadata': pdf_info, 'text': text}
# #     return text
    file.write(text)
    file.truncate()

def process_text():
    for i in get_files('.cermxml'):
        lxml_to_text(i)
    for i in get_files('.txt'):
        remove_tags(i)
    for i in get_files('.txt'):
        add_abstract(i)
    for i in get_files('.txt'):
        add_metadata(i)
process_text()
