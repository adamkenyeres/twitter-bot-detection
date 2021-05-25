from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, f1_score, confusion_matrix, roc_curve, auc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
class Evaulator():
    def __init__(self):
        pass
    
    def evaulate(self, y_pred, y, name):
        print(f'-----Evaulation of {name}-----')
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        print(f'Accuracy: {acc}')
        print(f'F1 Score: {f1}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
    
        print("Classification Report:")
        print(classification_report(y, y_pred))
    
        print("Confusion Matrix:")
        print(confusion_matrix(y, y_pred))
        self.plot_confusion_mtx(y_pred, y, name)
        self.plot_roc_curve(y_pred, y, name)
        return f1
    
    def plot_confusion_mtx(self, y, y_pred, name):
        print("Confusion Matrix:")
        mtx = confusion_matrix(y, y_pred)
        print(mtx[0][0])
        print(mtx)
        df = pd.DataFrame(mtx, index = ['Negative', 'Positive'],  columns = ['Negative', 'Positive'])        
        print(df.head())
        ax = sns.heatmap(df, annot=True)
        ax.set(xlabel='Predicted Class', ylabel='Actual Class')
        plt.title(f'Confusion Matrix for {name}')
        plt.show()

    def plot_roc_curve(self, y, y_pred, name):
        fpr, tpr, thresholds = roc_curve(y, y_pred)
        roc_auc = auc(fpr, tpr)
        print('fpr')
        print(fpr)
        print(tpr)
        
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (AUC = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver operating characteristic for {name}')
        plt.legend(loc="lower right")
        plt.show()

evaulator = Evaulator()
evaulator.evaulate([1,1,0, 1,0], [1,1,1, 0,1], 'Test')
