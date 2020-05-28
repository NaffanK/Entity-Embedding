import gensim
import gensim.downloader as api
import os
import gensim.models.keyedvectors as Word2Vec
import pyemd
import networkx as nx
import requests
import json
import math
from os import listdir
from os.path import isfile, join
from gensim.similarities import WmdSimilarity
from gensim.models.fasttext import FastText
from nltk.corpus import stopwords
from nltk import download
from pyemd import emd
from time import time

def load_w2v_model():
    w = gensim.models.KeyedVectors.load('C:/Users/Naffan/Desktop/Ryerson/Capstone/Wikipedia Embeddings/temp_model_Wiki', mmap='r')
    #w.init_sims(replace=True)
    return w

def get_url(URL, files, top):
    URL = URL + files[0]
    for file in files[1:]:
        URL = URL + "=&" + file
    URL = URL + "=&top=" + str(top) 
    return URL

def get_request(URL):
    return requests.get(url = URL)

def search_entity(data, corpus):
    doc2entity = {}
    for doc, entities in data.items():
        docname = doc.split("C:\\Users\\Naffan\\Desktop\\Ryerson\\Capstone\\tagme-doc-web\\app\\test_corpuses\\test_folder\\")[1]
        sentence = ''
        for entity in entities.keys():
            entity = entity.lower()
            entity = entity.replace(" ", "_")
            if ('e_' + entity) in corpus:
                if not sentence:
                    #print(('e_' + entity))
                    sentence = sentence + ('e_' + entity)
                else:
                    #print(('e_' + entity))
                    sentence = sentence + ' ' + ('e_' + entity)
        #print(sentence.split())
        doc2entity[docname] = sentence.split()
    return doc2entity

def graphEdge(doc1, doc2, distance):
    G.add_edge(doc1,doc2,weight = distance)

def displayGraph(G):
    for u,v,e in G.edges(data=True):
        print(u,v,e)

def filtered_graph(threshold):
    for u,v,e in G.edges(data=True):
        for edge in e.values():
            if(edge > threshold):
                G_filtered.add_edge(u,v, weight=edge)

def normalize(score, scores):
    normal = (score - min(scores))/(max(scores) - min(scores))
    return normal

def display_result(topic, B, scores):
    resultpath = "C:\\Users\\Naffan\\Desktop\\Ryerson\\Capstone\\trec_eval-master\\test\\Entity_Results2.test"
    with open(resultpath, 'a') as f:
        for doc1 in scores.keys():
            for doc2 in B.keys():
                if(doc1 == doc2):
                    output = topic + " " + doc1 + " " + str(normalize(B[doc1], B.values())) + " " + str(normalize(scores[doc2], scores.values()))
                    print(output)
                    #f.write(output+"\n")
    
def sort_scores(B):
    B_sorted = sorted(B.items(), key = lambda x: x[1], reverse= True)
    for pair in B_sorted:
        print(pair[0], ' ', pair[1]) 

start = time()

w2v_model=load_w2v_model()
corpus = list(w2v_model.vocab)

docpath = "C:\\Users\\Naffan\\Desktop\\Ryerson\\Capstone\\trec_eval-master\\test\\skipped.test"
#onlyfiles = [f for f in listdir(docpath) if isfile(join(docpath, f))]
URL = "http://localhost:5000/get_doc_entity/?"
documents = []
scores = {}
topic_count = ''
G = nx.Graph()
G_filtered = nx.Graph()
with open(docpath, 'r') as result_file:
    for line in result_file:
        line = line.split()
        #print(line[0],line[1],line[2])
        if not topic_count:
            topic_count = line[0]
            #print(topic_count)

        if not(not line):
            if(topic_count != line[0]):
                URL = get_url(URL, documents, 10)
                data = get_request(URL).json()
                URL = "http://localhost:5000/get_doc_entity/?"
                searched_entities = search_entity(data, corpus)
                distance_sum = 0
                edge_count = 0
                for docname1, entity_list1 in searched_entities.items():
                    for docname2, entity_list2 in searched_entities.items():
                        if docname1 == docname2:
                            continue

                        distance = w2v_model.wmdistance(entity_list1, entity_list2)
                        if(not math.isinf(distance)):
                            graphEdge(docname1, docname2, distance)
                            distance_sum = distance_sum + distance
                            edge_count = edge_count + 1
                
                threshold = distance_sum / edge_count
                filtered_graph(threshold)
                #displayGraph(G_filtered)
                B = nx.betweenness_centrality(G_filtered, normalized=False)
                display_result(topic_count,B,scores)
                #sort_scores(B)
                documents = []
                scores = {}
                documents.append(line[1])
                scores[line[1]] = float(line[2])
                topic_count = line[0]
                G = nx.Graph()
                G_filtered = nx.Graph()
            else:
                documents.append(line[1])
                #print(documents)
                scores[line[1]] = float(line[2])
                #print(scores)


print ('Program took %.2f seconds to run.' %(time() - start))
