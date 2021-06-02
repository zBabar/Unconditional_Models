#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import KFold

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,f1_score, recall_score, precision_score, average_precision_score
import re
import baselines_new as b
import baselines as b_old
import pickle


# In[2]:


def extract_report(cleaned_data,cleaned_study,train_index,test_index):
    
    
    
    X_train, X_test = np.array(cleaned_data)[train_index], np.array(cleaned_data)[test_index]
    report,idx=b.find_best_report_cider(reports=X_train)
    print(train_index[idx])
    
    
    
    return report,cleaned_study[train_index[idx]],b.all_scores(X_test,report)


# In[3]:


orig_data=pd.read_csv('../data/processed_data.csv')

print(orig_data.columns)

study=orig_data['study'].to_list()
data=orig_data['findings'].to_list()



cleaned_data=[]
cleaned_study=[]
pattern = '[0-9]'
for inst,s in zip(data[:5],study[:5]):

    if isinstance(inst, str) :
    #print(type(inst))
    #print(inst)

        inst=inst.lower()
        inst1= inst.replace(',', '').replace("'", "").replace('"', '').replace('.', '')
        inst1 = inst1.replace('&', 'and').replace('(', '').replace(")", "").replace('-', ' ').replace('___','')
        inst2=re.sub(pattern, '', inst1)
        
        inst3=re.sub("\n|\r", "", inst2)
        
        cleaned_study.append(s)
        cleaned_data.append(inst3)

#print(cleaned_data[4])
kf = KFold(n_splits=5)
kf.get_n_splits(cleaned_data)

ten_fold_result=[]

ten_reports=[]

b1,b2,b3,b4,M,R,C=[],[],[],[],[],[],[]



# for train_index,test_index in kf.split(cleaned_data):
#     print(train_index)

#     X_train, X_test = np.array(cleaned_data)[train_index], np.array(cleaned_data)[test_index]
    
   
#     report,best=extract_report(X_train)
#     #print(report,cleaned_study[best])
#     print(b_old.all_scores(X_test,report))
#     bleu,MET,ROU,CID=b_old.all_scores(X_test,report)
    
#     b1.append(bleu[0])
#     b2.append(bleu[1])
#     b3.append(bleu[2])
#     b4.append(bleu[3])
#     M.append(MET)
#     R.append(ROU)
#     C.append(CID)

# #    


import multiprocessing as mp

# Step 1: Init multiprocessing.Pool()

pool = mp.Pool(mp.cpu_count())


scores=pool.starmap(extract_report,[(cleaned_data,cleaned_study,train_index,test_index) for train_index,test_index in kf.split(cleaned_data)])

pool.close()    

#print(scores)
# print(np.mean(b1),np.mean(b2),np.mean(b3),np.mean(b4),np.mean(M),np.mean(R),np.mean(C))


# for score in scores:
#     print(score[1])

# with open('baseline_reports.p','wb') as f:
    
#     pickle.dump(scores,f)
    

with open('baseline_reports.p','rb') as f:
    
    scores=pickle.load(f)

for score in scores:
    print(score[1])

