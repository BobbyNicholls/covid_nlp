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

from utils.data_utils import import_5k_covid_toy_set
from utils.text_preprocessing_utils import normalise, tokenise
from utils.text_analysis_utils import get_cluster, get_common_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from networkx.convert_matrix import from_numpy_array
from networkx.algorithms.tree import maximum_spanning_edges
from wordcloud import WordCloud
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

raw_text_df = import_5k_covid_toy_set()

raw_documents = list(raw_text_df["snippet"])
processed_documents = [normalise(tokenise(document)) for document in raw_documents]
processed_documents = [" ".join(x) for x in processed_documents]

vectorizer = TfidfVectorizer()
tf_idf_matrix = vectorizer.fit_transform(processed_documents)
tf_idf_df = pd.DataFrame(
    tf_idf_matrix.toarray(), columns=vectorizer.get_feature_names()
)

# cosine_similarity_array = cosine_similarity(tf_idf_df)

kmeans = KMeans(n_clusters=10, random_state=0).fit(tf_idf_matrix.toarray())

raw_text_df["doc_cluster"] = tf_idf_df.apply(get_cluster, args=[kmeans], axis=1)

clustered_documents_df = pd.DataFrame(
    raw_text_df.groupby(["doc_cluster"])["snippet"]
    .transform(lambda x: " ".join(x))
    .drop_duplicates()
)

raw_clustered_documents = list(clustered_documents_df["snippet"])
processed_clustered_documents = [
    normalise(tokenise(document)) for document in raw_clustered_documents
]
processed_clustered_documents = [" ".join(x) for x in processed_clustered_documents]

vectorizer = TfidfVectorizer()
clustered_tf_idf_matrix = vectorizer.fit_transform(processed_clustered_documents)
clustered_tf_idf_df = pd.DataFrame(
    clustered_tf_idf_matrix.toarray(), columns=vectorizer.get_feature_names()
)

clustered_cosine_similarity_array = cosine_similarity(clustered_tf_idf_df)
for i in range(len(clustered_cosine_similarity_array)):
    clustered_cosine_similarity_array[i][i] = 0

G = from_numpy_array(clustered_cosine_similarity_array)
plt.figure(figsize=(7, 7))
nx.draw(G, node_size=200, node_color="y", with_labels=False)
plt.show()

mst_edges = maximum_spanning_edges(G)
mst = nx.Graph()
for edge in mst_edges:
    mst.add_edge(edge[0], edge[1], weight=1)

plt.figure(figsize=(7, 7))
nx.draw(mst, node_size=200, node_color="y", with_labels=True)
plt.show()

wc = WordCloud()
wc.generate(raw_clustered_documents[4])
plt.imshow(wc)

get_common_words(raw_clustered_documents[9])


