# Entity-Embedding
Program that uses Entities in documents and Word Mover's Distance to calculate distance

This program uses Luis Zugasti's work of the TAGME application [(found here)](https://github.com/luiszugasti/tagme-doc-web) and Gensim's Word Mover's Distance. The goal is to use the entities generated from the TAGME application, search the correct entities in a Wikipedia corpus and use the corresponding word vectors to calculate the "distance" between these entities. This acts as a sort of psuedo hyperlink structure for documents to create a graph and use it to determine which documents are most important in that network using graph centrality. This program uses Betweenness Centrality to rank the documents.

The program takes a list of pre-ranked documents (either from SDM, BM25, TF-IDF, or any other precision-focused algorithm) to re-rank the documents and achieve higher precision/recall scoress compared to baseline results. Scores are determined using the TREC evaluation software.
