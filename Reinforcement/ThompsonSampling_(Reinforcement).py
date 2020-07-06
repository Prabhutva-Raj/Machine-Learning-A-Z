"""Created on Sun May  3 21:39:18 2020"""

# Thompson Sampling (Reinforecement Learning)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

#------------------------------------------------------------------------------------

'''Implementing Thompson Sampling'''

import random
N = 10000
d = 10
ads_selected = []
total_reward = 0

#step1----
numbers_of_rewards_1 = [0] * d      # no of times ad i got reward 1 upto round n
numbers_of_rewards_0 = [0] * d      # no of times ad i got reward 0 upto round n

#step2 and step3----
for n in range (0,N):
    max_random = 0
    ad = 0
    for i in range (0,d):
        random_beta = random.betavariate(numbers_of_rewards_1[i]+1 , numbers_of_rewards_0[i]+1)
            
        if random_beta > max_random:
            max_random = random_beta
            ad = i
            
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        numbers_of_rewards_1[ad]+=1
    else:
        numbers_of_rewards_0[ad]+=1
    total_reward += reward
        
#------------------------------------------------------------------------------------

''' Visualizing the results'''
plt.hist(ads_selected)
plt.title('Histogram of ads selection')
plt.xlabel('Ads')
plt.ylabel('No of times each add was selected')
plt.show()
        