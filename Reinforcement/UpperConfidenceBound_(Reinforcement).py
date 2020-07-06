"""Created on Sat Apr 25 15:05:03 2020"""

# Upper Confidence Bound

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

#------------------------------------------------------------------------------------

'''Implementing UCB'''

import math
N = 10000
d = 10
ads_selected = []
total_reward = 0

#step1----
numbers_of_selections = [0] * d      # no of times ad i was selected upto round n
sums_of_rewards = [0] * d            # sum of rewards of the ad i upto round n

#step2 and step3----
for n in range (0,N):
    max_upper_bound = 0
    ad = 0
    for i in range (0,d):
        if (numbers_of_selections[i] > 0):
            average_reward = sums_of_rewards[i]/numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n+1) / numbers_of_selections[i])   # n+1 as in python the index starts with 0 
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
            
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
            
    ads_selected.append(ad)
    numbers_of_selections[ad] += 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] += reward
    total_reward += reward
        
#------------------------------------------------------------------------------------

''' Visualizing the results'''
plt.hist(ads_selected)
plt.title('Histogram of ads selection')
plt.xlabel('Ads')
plt.ylabel('No of times each add was selected')
plt.show()
        
        
    
