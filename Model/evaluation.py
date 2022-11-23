
import numpy as np
import  math
from scipy.special import softmax

try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import matthews_corrcoef, f1_score, precision_recall_fscore_support, precision_score, recall_score, roc_auc_score,roc_curve

    _has_sklearn = True
except (AttributeError, ImportError):
    _has_sklearn = False


from sklearn.metrics import precision_recall_curve, accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import auc as AUC


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


sigmoid_v = np.vectorize(sigmoid)


def multi_task_metrics(out_by_task, labels_by_task):
    acc = 0
    results={}
    accuracies=0
    for task in out_by_task.keys():
        out, labels = np.array(out_by_task[task]), np.array(labels_by_task[task])
        outputs = np.argmax(out, axis=1)
        #acc += np.sum(outputs == labels)
        accuracy = accuracy_score(y_true=labels, y_pred=outputs)
        accuracies += accuracy
        probas_pred = np.max(softmax(out, axis=1),axis=1)
        #precision, recall, thresholds = precision_recall_curve(y_true=labels, probas_pred=probas_pred, pos_label=1)
        #roc_auc=roc_auc_score(labels, probas_pred)
        #auc_pr = AUC(recall, precision)
        #results[task] = {'auc_pr': auc_pr,'roc_aux':0, 'accuracy_score':accuracy}
    return accuracies/len(out_by_task)

def multi_label_metrics(preds, labels, label2id):
    assert len(preds) == len(labels)
    results = {}
    fpr_tpr={}
    roc_auc = []
    pr_auc=[]
    data_gt = np.array([l.tolist() for l in labels])
    data_pd = np.array([p.tolist() for p in preds])
    for label_name, i in label2id.items():
        auc = roc_auc_score(data_gt[:, i], data_pd[:, i])
        fpr, tpr, thresholds = roc_curve(data_gt[:, i], data_pd[:, i])
        #probas_pred = np.max(softmax(data_pd[:, i], axis=1), axis=1)
        #precision, recall, thresholds = precision_recall_curve(y_true=labels, probas_pred=probas_pred)
        #auc_pr = AUC(recall, precision)
        results[label_name] = {'auc': auc}
        fpr_tpr[label_name] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
        roc_auc.append(auc)
        #pr_auc.append(auc_pr)
    roc_auc = np.mean(roc_auc)
    #pr_auc=np.mean(pr_auc)
    results["auc"] = roc_auc
    preds = [(p >= 0.5).astype(int) for p in preds]
    results["accuracy"] = accuracy_score(labels,preds)
    #results["pr_auc"] = pr_auc
    return results, roc_auc , fpr_tpr

def label_metrics(preds, labels):
    assert len(preds) == len(labels)
    results = {}
    data_gt = np.array([l for l in labels])
    data_pd = np.array([p for p in preds])

    auc = roc_auc_score(data_gt, data_pd)
    fpr, tpr, thresholds = roc_curve(data_gt, data_pd)
    #probas_pred = np.max(softmax(data_pd[:, i], axis=1), axis=1)
    #precision, recall, thresholds = precision_recall_curve(y_true=labels, probas_pred=probas_pred)
    #auc_pr = AUC(recall, precision)
    fpr_tpr = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
    #pr_auc=np.mean(pr_auc)
    results["auc"] = auc
    preds = [(p >= 0.5).astype(int) for p in preds]
    results["accuracy"] = accuracy_score(labels,preds)
    #results["pr_auc"] = pr_auc
    return results, auc , fpr_tpr

