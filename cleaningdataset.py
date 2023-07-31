import numpy as np 
import pandas as pd 
 


df = pd.read_csv("adult.data.Eesha.2.csv") 
 
 
 
df[' Workclass'] = df[' Workclass'].replace(' ?', np.nan) 
df[' Occupation'] = df[' Occupation'].replace(' ?', np.nan) 
 
 
 
df.to_csv("adult.data.Eesha.2.csv", index = False) 