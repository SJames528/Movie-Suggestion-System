import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.spatial.distance import pdist, squareform

parent_file = os.path.dirname(os.getcwd())


def similarity_jacc(list1,list2):
    list1 = list1[0]
    list2 = list2[0]
    intersection = [cast for cast in list1 if cast in list2]
    union = list1 + [cast for cast in list2 if cast not in list1]
    index = len(intersection) / len(union)
    return index

def similarity_overlap(list1,list2):
    list1 = list1[0]
    list2 = list2[0]
    intersection = [cast for cast in list1 if cast in list2]
    index = len(intersection) / min(len(list1),len(list2))
    return index



## load in entire netflix dataset
movie_data = pd.read_csv(parent_file + "/data/netflix_titles.csv")

## extract only show id and cast, which we will use to create a basic heatmap of similarity between titles
cast_data = movie_data[["show_id","cast"]]

## remove entries with NaN in the cast column
cast_data = cast_data.dropna()


cast_data["cast"] = cast_data["cast"].map(lambda cast_list: cast_list.split(", "))

cast_data_short = cast_data["cast"][0:250]

cast_array = cast_data_short.values.reshape(len(cast_data_short),1)
cast_similarity_jacc = squareform(pdist(cast_array, similarity_jacc))
cast_similarity_overlap = squareform(pdist(cast_array, similarity_overlap))

sns.heatmap(cast_similarity_jacc)
plt.show()

sns.lineplot(x = range(0,250), y = cast_similarity_jacc[50])
plt.show()
