"""
How to use seaborn to find the correlation?
"""

This line could be used to find the correlation matrix:
print (X.corr(method="pearson"))

This line could be used to draw the correlation heat map:
sns.heatmap(X.corr(method="pearson"))

You could select one of the three methods:
method=PEARSON, KENDALL, SPEARMAN

https://www.statisticssolutions.com/correlation-pearson-kendall-spearman/
