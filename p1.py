# https://www.kaggle.com/datasets/hesh97/titanicdataset-traincsv
# 1.PROBABILITY
# a.Calculate simple Probability
import pandas as pd
df=pd.read_csv(r'C:\Users\user\Downloads\train (1).csv')
probability_event=df['Survived'].value_counts()/len(df['Survived'])
print(probability_event)

#b. Application of probability distribution, dropping missing values and plotting histogram
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
titanic_data=df.dropna(subset=['Age'])
plt.hist(df['Age'],bins=30,density=True,alpha=0.5,color='g',label='Age Distribution')
plt.title('Age Distribution')

# normal distribution of the data
mu,std=norm.fit(titanic_data['Age'])
xmin,xmax=plt.xlim()
x=np.linspace(xmin,xmax,100)
p=norm.pdf(x,mu,std)
plt.plot(x,p,'k',linewidth=2)

# display histogram plot
plt.hist(df['Age'],bins=30,density=True,alpha=0.5,color='g',label='Age Distribution')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()
plt.show()
