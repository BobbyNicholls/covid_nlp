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

from utils.data_utils import import_toy_set
from utils.text_preprocessing_utils import normalise, tokenise
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import pandas as pd

raw_text_df = import_toy_set()

raw_documents = list(raw_text_df["snippet"])

words = ["this", "is", "many", "sentences", "with", "0", "context"]

processed_documents = [normalise(tokenise(document)) for document in raw_documents]
processed_documents = [" ".join(x) for x in processed_documents]


vectorizer = TfidfVectorizer()
tf_idf_matrix = vectorizer.fit_transform(processed_documents)

tf_idf_df = pd.DataFrame(
    tf_idf_matrix.toarray(), columns=vectorizer.get_feature_names()
)

cosine_similarity_array = cosine_similarity(tf_idf_df)

kmeans = KMeans(n_clusters=2, random_state=0).fit(cosine_similarity_array)

kmeans.cluster_centers_

kmeans.predict()

