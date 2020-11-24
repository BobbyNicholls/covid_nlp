"""
Module for functions that analyse text
"""

def get_cluster(row, kmeans):
    return kmeans.predict(row.values.reshape(1, -1))[0]
