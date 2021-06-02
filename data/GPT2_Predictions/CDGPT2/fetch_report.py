import pandas as pd

import numpy as np
#

######################################### SAT Generated and MRA Generated



data=pd.read_csv('score1.csv',encoding='latin-1')

print(data.columns)

#
#
#
d=data['ref']


print(d[0][0])

data1=[]

for i in d:
    print(type(i))
    i=i.replace('.','')
    i = i.replace('[', '')
    i = i.replace(']', '')
    #i=i.replace('u ','')
    i = i.replace("u'", '')
    i = i.replace("'", '')
    i=str(i)
    data1.append(i.strip())


print(data1)


data['cap']=data1


data['cap'].to_csv('score_ref_cleaned1.csv',header=False,index=False)




################################################ To combine report-id and new labels
#
#
# data_labeled_test=pd.read_csv('labeled_reports_test.csv')
#
# #
# # data_labeled_test['report_id']=data['report_id']
# #
# # #data_test=data.merge(data_labeled,left_index=True,right_index=True)
# #
# # data_labeled_test.to_csv('test_labeled.csv')
#
#
# data_labeled_train=pd.read_csv('labeled_reports_train.csv')
# data=pd.read_json('train.json')
#
# data_labeled_train['report_id']=data['report_id']
#
# #data_test=data.merge(data_labeled,left_index=True,right_index=True)
#
# data_labeled_train.to_csv('train_labeled.csv')
#
# data_labeled=pd.concat([data_labeled_train,data_labeled_test])
#
# data_labeled.to_csv('data_labeled.csv')


######################################################################### to combine new labels with impression_findings based data based on report_id


# train=pd.read_json('train.json')
#
# test=pd.read_json('word_sent_tags_14.json')
#
#
# Labels=pd.read_csv('data_labeled.csv')
#
#
# train_tags=train.merge(Labels,on='report_id')
#
# train_X=train_tags[['report_id','caption']]
# train_Y=train_tags[['No Finding',
#        'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion',
#        'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
#        'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
#        'Support Devices']]
#
# np.savez('train_data',x=train_X,y=train_Y)
#
# train_tags.to_csv('train_data.csv',index=False)
#
# test_tags=test.merge(Labels,on='report_id')
#
#
# test_Y=test_tags[['No Finding',
#        'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion',
#        'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
#        'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
#        'Support Devices']]
#
# print(test_Y.shape)
# np.savez('test_data',x=test,y=test_Y)
#
# test_tags.to_csv('test_data.csv',index=False)

