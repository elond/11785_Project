import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast


data= pd.read_csv("all_results.csv")
df = pd.DataFrame(data, columns=['Dataset', 'NetworkName', 'Encoding', '1st Hidden Units', '2nd Hidden Units', '3rd Hidden Units', 'Window Size', 'Padding', 'Batch Size', 'LR', 'Epochs', 'ML_Dataset', \
                                 'True Positives', 'False Positives', 'True Negatives', 'False Negatives', 'Accuracy', 'Recall', 'Precision', 'Specificity', \
                                 'F1 score', 'AUROC', 'Test Run Time (s)', 'Train time (s)'

])

DATASET = "Test"
metrics = ['Accuracy', 'Recall', 'Precision', 'Specificity', 'F1 score', 'AUROC']
datasets = ['GenBank', 'RefSeq']
encodings = ['One_hot', 'CAT']
models = ['CNNNet', 'LSTMNet', 'CNNLSTMNet']

for i, metric in enumerate(metrics):
    
    # Create grid with 4 plots
    #fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig, axes = plt.subplots(2, 2)
    axes[0][0].set_title("GenBank", fontsize=10)
    axes[0][1].set_title("RefSeq", fontsize=10)
    fig.suptitle('{} variation in encoding and dataset'.format(metric))        
    i = 0
    for encoding in encodings:
        j = 0
        for dataset in datasets:
            Y = []
            for model in models:
                l = len(model)
                rslt_df = df.loc[ ( df['NetworkName'].str[:l] == model) & (df['Encoding']==encoding) & \
                                         (df['Dataset'] ==dataset) & (df['ML_Dataset'] == DATASET) ]
                if metric != "AUROC":
                    rslt_df[metric] = rslt_df[metric].apply(lambda x:np.average(ast.literal_eval(' '.join(x.split()).replace(' ',',')) ))
                metric_value = rslt_df[metric].max()
                Y.append(metric_value)
            
            if encoding == 'CAT':
                if dataset == 'GenBank':
                    axes[i][j].bar(['CNN', 'LSTM', 'CNNLSTM'], Y, color='tab:orange', label='CAT')
                else:
                    axes[i][j].bar(['CNN', 'LSTM', 'CNNLSTM'], Y, color='tab:orange')
            else:
                if dataset == 'GenBank':
                    axes[i][j].bar(['CNN', 'LSTM', 'CNNLSTM'], Y, label='One Hot')
                else:
                    axes[i][j].bar(['CNN', 'LSTM', 'CNNLSTM'], Y)

            lim_min = np.min(Y) - 0.1 if np.min(Y) - 0.1 > 0 else 0
            lim_max = np.max(Y) + 0.1 if np.max(Y) + 0.1 < 1.0 else 1.0
            axes[i][j].set_ylim([lim_min, lim_max])
            
            j += 1
        
        i += 1
    
    #for ax in fig.get_axes():
    #    ax.label_outer()
    fig.legend()         
    plt.savefig("{}_plots.png".format(metric))
