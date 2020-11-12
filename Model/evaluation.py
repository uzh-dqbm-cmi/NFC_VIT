
import numpy as np

try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import matthews_corrcoef, f1_score, precision_recall_fscore_support, precision_score, recall_score, roc_auc_score

    _has_sklearn = True
except (AttributeError, ImportError):
    _has_sklearn = False


from sklearn.metrics import precision_recall_curve, accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import auc as AUC

def multi_task_metrics(out_by_task, labels_by_task):
    acc = 0
    results={}
    accuracies=0

    for task in out_by_task.keys():
        out, labels = np.array(out_by_task[task]), np.array(labels_by_task[task])
        outputs = np.argmax(out, axis=1)
        #acc += np.sum(outputs == labels)
        accuracy=accuracy_score(y_true=labels, y_pred=outputs)
        accuracies+=accuracy
        probas_pred = np.max(out,axis=1)
        precision, recall, thresholds = precision_recall_curve(y_true=labels, probas_pred=probas_pred, pos_label=1)
        #roc_auc=roc_auc_score(labels, probas_pred)
        auc_pr = AUC(recall, precision)
        results[task] = {'auc_pr': auc_pr,'roc_aux':0, 'accuracy_score':accuracy}
    #print(accuracies/len(out_by_task), acc / (len(out_by_task)*len(out_by_task[task])) )
    return accuracies/len(out_by_task), results

