# -*- coding: utf-8 -*-
"""
@author: morfoula
"""

"""
A standalone script to download and parse edgar 10k MDA section
"""

import pandas as pd

# Load data directly from the local CSV file
print("Loading Final_Dataset.csv...")
df = pd.read_csv('Final_Dataset.csv')

# If you need to do further processing, continue below

MDA_Dataset=pd.DataFrame(df)
MDA_Dataset=MDA_Dataset.dropna()
MDA_Dataset=MDA_Dataset.iloc[0:352,:]


section_text={'Index': [], 'Data' : []}

for i in range(352):
    f=MDA_Dataset.iloc[i,5]
    section_text['Index'].append(MDA_Dataset.iloc[i,0])
    section_text['Data'].append(f) # Assuming 'f' is the text content for now
    
  
section_text=pd.DataFrame(section_text)


final=pd.merge(MDA_Dataset,section_text, on=['Index'])
final.to_excel(r'C:\Users\morfo\Desktop\10K-MDA-Section-master\Dataset.xlsx',index=False)