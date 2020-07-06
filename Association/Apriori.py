"""Created on Wed Apr 22 17:01:48 2020"""

# Apriori

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("D:\\MachineLearning\\MachineLearning A to Z\\Machine Learning A-Z New\\Part 5 - Association Rule Learning\\Section 28 - Apriori\\Apriori_Python\\Market_Basket_Optimisation.csv", header = None)

#------------------------ but apriori wants list of list ---------------------------

transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])
    
#------------------------------------------------------------------------------------
    
'''training apriori on the dataset'''
from apyori import apriori    # apyori is a self made package
rules = apriori(transactions, min_support=0.003, min_confidence=0.20, min_lift=0.3, min_length=2)
# min_support = 3timesADay*7days / 7500
    
#------------------------------------------------------------------------------------
    
'''visualizing the results'''
results = list(rules)
# we wont sort the rules because here they will be sorted on basis of combination of support,confidence and lift ; by the apyori itself

