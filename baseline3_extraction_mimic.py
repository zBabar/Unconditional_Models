from sklearn.model_selection import KFold

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,f1_score, recall_score, precision_score, average_precision_score


def extract_report(train_data):

    n=train_data.shape[0]
    #print(type(train_data))
    accuracy=[]

    #print(train_data)


    for index in range(0,n-1):
        if (index%10000==0):
            print(index)

        pred=train_data.iloc[index,:]
        # pred = pd.DataFrame(pred).T
        # pred = pd.concat([pred] * n, ignore_index=True)
        #
        #
        #
        # #print(pred.shape,train_data.shape)
        # pred=np.mean(pred,axis=0)

        pred=pred*1.0
        #accuracy.append(accuracy_score(train_data,pred))
        comp=train_data* pred + (1 - train_data) * (1 - pred)
        accu = np.mean(comp,axis=1)

        accuracy.append(np.mean(accu))
        #print(accuracy)




    return accuracy


orig_data=pd.read_csv('mimic-cxr-2.0.0-chexpert.csv')

data=orig_data.iloc[:,2:]
print(data.shape)

data=orig_data.iloc[:,2:]

data=data.fillna(0)

data = data.replace(-1, 0)

kf = KFold(n_splits=10)
kf.get_n_splits(data)

ten_fold_result=[]

ten_reports=[]
for train_index,test_index in kf.split(data):


    X_train, X_test = data.iloc[train_index,:], data.iloc[test_index,:]
    X_train=X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    accuracy=extract_report(X_train)

    report_id=np.argmax(accuracy)
    #print(accuracy)

    best_idx=train_index[report_id]
    print(best_idx,orig_data.iloc[best_idx,:2])
    ten_reports.append(report_id)
    n=X_test.shape[0]
    r=pd.DataFrame(X_train.iloc[report_id,:]).T

    pred = r * 1.0
    # accuracy.append(accuracy_score(train_data,pred))
    comp = X_test * pred + (1 - X_test) * (1 - pred)
    accu = np.mean(comp, axis=1)

    print('accuracy1:', np.mean(accu))

    pred1 = pd.concat([r] * n, ignore_index=True)



    #accu1=accuracy_score(X_test,pred)

    #print('accuracy:', accu1)
    ten_fold_result.append(np.mean(accu))

    # prec_macro=np.sum(X_test*pred,axis=0)/(pred*X_test.shape[0])
    # #print('manual prec_macro classwise',prec_macro)
    # print('manual prec_macro', np.mean(prec_macro))
    #
    # prec_micro = np.sum(X_test * pred) / (np.sum(pred) * X_test.shape[0])
    # print('manual prec_micro', prec_micro)
    #
    # rcl_macro = np.sum(X_test * pred, axis=0) /np.sum(X_test,axis=0)
    # #print('manual rcl_macro classwise', rcl_macro)
    # print('manual rcl_macro', np.mean(rcl_macro))
    #
    # rcl_micro = np.sum(X_test * pred) /np.sum(X_test)
    # print('manual rcl_micro', rcl_micro)


    print('precision_class wise', average_precision_score(X_test, pred1, average=None))
    print('precision_macro',average_precision_score(X_test,pred1, average='macro'))
    print('recall_macro',recall_score(X_test,pred1, average='macro'))
    print('precision_micro',average_precision_score(X_test,pred1, average='micro'))
    print('recall micro',recall_score(X_test,pred1, average='micro'))



print('ten best report indices',ten_reports)

print(np.argmax(ten_fold_result))

print('average accuracy',np.mean(ten_fold_result))







