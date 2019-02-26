import numpy as np
import pandas as pd

df=pd.read_table("C:/Users/wrm/Desktop/CB.txt",encoding="ANSI",header=None,sep=",")

from sklearn.cluster import DBSCAN
x=df.drop([0],axis=1)
#print (x)
clat=DBSCAN(eps=500).fit(x)
print (clat.labels_)

"""
This algorithm performed very bad for this clustering problem. K-means is much
better for the problem.
"""
