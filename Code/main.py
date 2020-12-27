import os
import sys
import  numpy as np
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split, KFold
from util import gpt, bert, finebert, load_finebert
import  pandas as pd
def test_kfold(true_labels, predictions):
    kf = KFold(n_splits = 10)
    accuracy = []
    f1=[]
    for train_index, test_index in kf.split(true_labels):
        true_labels_split = true_labels[test_index]
        predictions_split = predictions[test_index]
        accuracy.append(np.sum(predictions_split == true_labels_split)/len(true_labels_split))
        # f1.append(classification_report(true_labels_split, predictions_split, output_dict= True)['macro avg']['f1-score'])

    print(f'mean += std: {round(np.mean(accuracy), ndigits=5)*100} +- {round(np.std(accuracy), ndigits=5)*100}')
    # print(f'mean += std: {round(np.mean(f1), ndigits=5)*100} +- {round(np.std(f1), ndigits=5)*100}')
    return accuracy, f1

def compare():
    

    print("GPT***************************")
    true_labels, predictions = gpt()
    accuracy_gpt, _ = test_kfold(true_labels, predictions)
    print(accuracy_gpt)
    print("BERT***************************")
    true_labels, predictions = bert()
    accuracy_bert, _ = test_kfold(true_labels, predictions)
    print(accuracy_bert)
    print("FINEBERT***************************")
    true_labels, predictions = load_finebert()
    accuracy_finebert, _ = test_kfold(true_labels, predictions)
    print(accuracy_finebert)

    testResults=[]
    testResults.append(accuracy_gpt)
    testResults.append(accuracy_bert)
    testResults.append(accuracy_finebert)

    from scipy import stats
    tdata=testResults

    name=['GPT','BERT','Fine_tuned_BERT']
    ttest_df=pd.DataFrame(index=name,columns=name)
    for i in range(len(tdata)):
        for j in range(len(tdata)):
            ttest_df.iloc[i,j]=stats.ttest_ind(tdata[i],tdata[j])[1] # [0] T-value [1] P-values


    print(ttest_df)





def main():

    mode = sys.argv[1]

    if mode == 'gpt2':
        gpt()

    if mode == 'bert':
        bert()

    if mode == 'finebert':
        finebert()
    if mode == 'load_finebert':
        load_finebert()
    if mode == 'compare':
        compare()
        




if __name__ == "__main__":
    main()