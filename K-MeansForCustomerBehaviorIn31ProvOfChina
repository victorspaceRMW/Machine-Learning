import numpy as np
import pandas as pd

df=pd.read_csv("C:/Users/wrm/Desktop/CustomerBehavior.csv",names=["省份","食品","衣着","家庭用品以及服务","医疗保健","交通与通讯","娱乐与教育","居住","杂项"],encoding="ANSI")

from sklearn.cluster import KMeans
X=df.drop(columns=["省份"])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
prov=df["省份"]
#print (list(df2))
c={"Prov":list(prov),"Label":kmeans.labels_}
result=pd.DataFrame(c)
print (result)
