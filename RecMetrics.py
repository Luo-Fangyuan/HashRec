import math
from sklearn.metrics import roc_auc_score
import numpy as np

'''
label and pred are both nested lists, where each sub-list represents the label list or prediction list of one user.
'''


class Metric(object):
    def __init__(self, label, pred, K):
        self.label = label
        self.pred = pred
        self.K = K

    def hit_ratio(self):
        hit_num = 0
        for ind in range(len(self.label)):
            rank = np.argsort(-np.array(self.pred[ind]))
            sorted_label = [np.array(self.label[ind])[i] for i in rank]
            if np.sum(sorted_label[:min(self.K, len(self.label))]) != 0:
                hit_num += 1
        return hit_num / len(self.label)


    def precision(self):
        precision_value = 0
        for ind in range(len(self.label)):
            rank = np.argsort(np.argsort(-np.array(self.pred[ind])))
            sorted_label = [np.array(self.label[ind])[rank[i]] for i in rank]
            rank = np.asfarray(sorted_label)[:min(self.K, len(self.label))]
            precision_value += np.sum(rank) / self.K
        return precision_value / len(self.label)

    def recall(self):
        recall_value = 0
        for ind in range(len(self.label)):
            rank = np.argsort(-np.array(self.pred[ind]))
            sorted_label = [np.array(self.label[ind])[i] for i in rank]
            rank = np.asfarray(sorted_label)[:self.K]
            recall_value += np.sum(rank) / np.sum(np.array(self.label[ind]))
        return recall_value / len(self.label)

    def accuracy(self, thre=0):
        accuracy_value = 0
        count = 0
        for ind in range(len(self.label)):
            self.pred[ind] = np.where(np.array(self.pred[ind]) > thre, 1, 0)
            accuracy_value += np.sum(np.array(self.pred[ind]) == np.array(self.label[ind]))
            count += len(self.pred[ind])
        return accuracy_value / count

    def mae(self):
        error = 0
        count = 0
        for ind in range(len(self.label)):
            error += np.sum(abs((np.array(self.label[ind]) - np.array(self.pred[ind]))))
            count += len(self.label[ind])
        if count==0:
            return error
        return round(error/count,5)

    def rmse(self):
        error = 0
        count = 0
        for ind in range(len(self.label)):
            error += np.sum((np.array(self.label[ind]) - np.array(self.pred[ind]))**2)
            count += len(self.label[ind])
        if count==0:
            return error
        return round(math.sqrt(error/count),5)

    def ndcg(self):
        sum_NDCG = 0
        for ind in range(len(self.label)):
            lrank = np.array(self.label[ind]).argsort()[::-1]
            idcg_value = 0
            dcg_value = 0
            for k in range(min(self.K, len(self.label[ind]))):
                idcg_value += (2 ** np.array(self.label[ind])[lrank[k]] - 1) / math.log2(k + 2)
            prank = np.array(self.pred[ind]).argsort()[::-1]
            for k in range(min(self.K, len(self.label[ind]))):
                dcg_value += (2 ** np.array(self.label[ind])[prank[k]] - 1) / math.log2(k + 2)
            sum_NDCG += dcg_value / idcg_value
        return round(sum_NDCG / len(self.pred),5)

    def map(self):
        sum_prec = 0
        for ind in range(len(self.label)):
            rank = np.argsort(-np.array(self.pred[ind]))
            sorted_label = [np.array(self.label[ind])[i] for i in rank]
            precision_value = 0
            for i in range(min(self.K, len(self.label[ind]))):
                if self.label[ind][i] == 1:
                    precision_value += np.sum(sorted_label[:i]) / (i + 1)
            sum_prec += precision_value / np.sum(sorted_label[:min(self.K, len(self.label[ind]))])
        return sum_prec / len(self.label)

    def auc(self):
        sum_AUC = 0
        for user in range(len(self.label)):
            sum_AUC += roc_auc_score(self.label[user], self.pred[user])
        return float(sum_AUC) / len(self.label)

    def mrr(self):
        sum_MRR = 0
        for ind in range(len(self.label)):
            rank = np.argsort(np.argsort(-np.array(self.pred[ind])))
            sum_MRR += np.max(np.array(self.label[ind]) / (rank + 1))
        return float(sum_MRR) / len(self.label)






