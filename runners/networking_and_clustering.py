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

from utils.data_utils import import_reddit10k
from utils.network_utils import get_max_spanning_tree
from utils.text_preprocessing_utils import normalise, tokenise
from utils.text_analysis_utils import get_cluster, get_common_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from networkx.convert_matrix import from_numpy_array
from wordcloud import WordCloud
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

raw_text_df = import_reddit10k()

TEXT_FIELD = "body"
N_CLUSTERS = 5

raw_documents = list(raw_text_df[TEXT_FIELD])
processed_documents = [normalise(tokenise(document)) for document in raw_documents]
processed_documents = [" ".join(x) for x in processed_documents]

vectorizer = TfidfVectorizer()
tf_idf_matrix = vectorizer.fit_transform(processed_documents)
tf_idf_df = pd.DataFrame(
    tf_idf_matrix.toarray(), columns=vectorizer.get_feature_names()
)

# cosine_similarity_array = cosine_similarity(tf_idf_df)

kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0).fit(tf_idf_matrix.toarray())

raw_text_df["doc_cluster"] = tf_idf_df.apply(get_cluster, args=[kmeans], axis=1)

clustered_documents_df = pd.DataFrame(
    raw_text_df.groupby(["doc_cluster"])[TEXT_FIELD]
    .transform(lambda x: " ".join(x))
    .drop_duplicates()
)

raw_clustered_documents = list(clustered_documents_df[TEXT_FIELD])
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

mst = get_max_spanning_tree(G)
plt.figure(figsize=(7, 7))
nx.draw(mst, node_size=200, node_color="y", with_labels=True)
plt.show()

wc = WordCloud()
wc.generate(raw_clustered_documents[3])
plt.imshow(wc)

get_common_words(raw_clustered_documents[0])

cluster_count = raw_text_df["doc_cluster"].value_counts()
plt.figure(figsize=(10, 5))
sns.barplot(cluster_count.index, cluster_count.values, alpha=0.8)
plt.title("Cluster counts")
plt.ylabel("Number of Occurrences", fontsize=12)
plt.xlabel("Cluster", fontsize=12)
plt.show()
