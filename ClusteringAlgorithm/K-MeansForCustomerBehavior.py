import numpy as np
import pandas as pd

df=pd.read_csv("C:/Users/wrm/Desktop/CustomerBehavior.csv",names=["省份","食品","衣着","家庭用品以及服务","医疗保健","交通与通讯","娱乐与教育","居住","杂项"],encoding="ANSI")
"""
在这一行中，我们读入中国31个省的居民在食品，衣着，家庭用品以及服务等领域的平均消费数目。需要注意的是encoding方法需要使用ANSI。
"""

from sklearn.cluster import KMeans
"""
从cluster中引入kmeans算法。在这个题目中我们主要控制的变量是n_cluster的数目。也就是需要将这31个省的居民的消费情况，归为几个(n_cluster)类。假如n_cluster=2,
那么就是归为两类。同理，假如n_cluster=3,那么就是归为3类。以此类推。从经济学的角度自然也可以有不同的解释。如发达地区-欠发达地区。非常发达的地区-一般发达的
地区以及欠发达的地区。
"""
X=df.drop(columns=["省份"])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
prov=df["省份"]
#print (list(df2))
c={"Prov":list(prov),"Label":kmeans.labels_}
result=pd.DataFrame(c)
print (result)
