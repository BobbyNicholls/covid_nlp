"""
The overview:
The point of this is to try to bring in some text and cluster it in to topics, we will then  take those topics and
consider each to be a node in a network and build up a graphical relationship between the nodes. This could be used to
visualise how topics trend over time, the sentiment around them and their relationship to economic factors.

The detailed version:
Step 1: import the data

Step 2: do some routine pre-processing,
    lemmatisation,
    stop word removal,
    tokenisation..etc

Step 3: Turn this corpus in to a tf-idf (term frequency inverse document frequency) matrix. This deals very well with
    long documents

Step 4: Produce a matrix of relationships between each document using the cosine similarity of the vectors.

Step 5: Cluster these documents, initially for simplicity using out of the box k-means in to K clusters.

Step 6: Each cluster to be turned in to a super-document and these to be related in the same manner.

Step 7: A network is drawn and visualised, a max weighted spanning tree drawn and visualised

"""


import pandas as pd