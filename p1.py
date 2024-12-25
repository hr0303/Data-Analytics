# https://www.kaggle.com/datasets/hesh97/titanicdataset-traincsv
# 1.PROBABILITY
# a.Calculate simple Probability
import pandas as pd
df=pd.read_csv(r'C:\Users\user\Downloads\train (1).csv')
probability_event=df['survived'].value_counts()/len(df['survived'])
print(probability_event)

#b. Application of probability distribution, dropping missing values and plotting histogram
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
titanic_data=df.dropna(subset=['age'])
plt.hist(df['age'],bins=30,density=True,alpha=0.5,color='g',label='age Distribution')
plt.title('age Distribution')

# normal distribution of the data
mu,std=norm.fit(titanic_data['age'])
xmin,xmax=plt.xlim()
x=np.linspace(xmin,xmax,100)
p=norm.pdf(x,mu,std)
plt.plot(x,p,'k',linewidth=2)

# display histogram plot
plt.hist(df['age'],bins=30,density=True,alpha=0.5,color='g',label='age Distribution')
plt.title('age Distribution')
plt.xlabel('age')
plt.ylabel('frequency')
plt.legend()
plt.show()
