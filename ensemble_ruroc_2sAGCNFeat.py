import argparse
import pickle
from scipy.special import softmax
import numpy as np
from tqdm import tqdm
import os.path
from sklearn.metrics import f1_score,roc_auc_score,roc_curve,confusion_matrix,classification_report
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize
from sklearn import metrics
# from yellowbrick.classifier import ROCAUC
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
import matplotlib.colors as mcolors
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix, classification_report, auc

import pandas as pd
n_class=6
if os.path.isfile('alpha_t.npy') :
    alpha_t = np.load('alpha_t.npy')
    print(alpha_t)
    alpha_t = alpha_t.tolist()
else:
    alpha_t=[[0.2, 0.0, 0.0, 0.4, 0.4] , [0.2, 0.0, 0.0, 0.4, 0.4] , [0.3, 0., 0.0, 0.3, 0.3] , [0.2, 0.2, 0.2, 0.2, 0.2]]
alpha_t =[ [0.2, 0.2, 0.2, 0.2, 0.2]]

def store_df(data_type,res,run):
    #       NNval NNTest
    #  params.seed
    method_list = ['macro ROC OVR','weighted ROC OVR','macro ROC OVO','weighted ROC OVO','micro AUPRC', 'samples AUPRC', 'weighted AUPRC', 'macro AUPRC']

    if not os.path.isfile('compare_AUC_2sAGCNFeat.csv'):
        types = ['val','test']
        col_list =['RUNS']
        for mth in method_list:
            for exp_t in types:
                col_list.append(mth+"_"+exp_t)


        df = pd.DataFrame(columns=col_list)
        # df['Seed'] = np.arange(10)
        df.to_csv('compare_AUC_2sAGCNFeat.csv',index=False)
    # import pdb;pdb.set_trace()

    # import pdb;pdb.set_trace()
    df_full = pd.read_csv('compare_AUC_2sAGCNFeat.csv',index_col='RUNS')
    # df_full.loc[(seed_val),'Seed']= seed_val
    cols=[]
    for mth in method_list:
        cols.append(mth+"_"+data_type)
    # import pdb;pdb.set_trace()
    df_full.loc[run,cols] = res

    # df_full.loc[run,method_name+"_"+data_type] = res
    # df_full.loc[run,method_name+"_Test"] = test_res
    df_full.to_csv('compare_AUC_2sAGCNFeat.csv',index=True)

    
def cross_entropy(y,y_pre):
  y_=[0]*n_class
  y_[y] = 1
  y=y_
#   print(y,y_pre)
  loss=-np.sum(y*np.log(y_pre))
  return loss/float(y_pre.shape[0])
def find_max(acc,norm_acc,max_acc,max_nacc,max_acc_wt,max_nacc_wt,alpha):
    if acc>max_acc:
        max_acc=acc
        max_acc_wt = alpha
    if norm_acc>max_nacc:
        max_nacc=norm_acc
        max_nacc_wt=alpha
    return max_acc,max_acc_wt,max_nacc,max_nacc_wt

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', default='kinetics', choices={'kinetics', 'ntu/xsub', 'ntu/xview'},
                    help='the work folder for storing results')
parser.add_argument('--alpha', default=1, help='weighted summation')
parser.add_argument('--test', default=1, help='test or val data')
parser.add_argument('--run', default='RUN1', help='RUN value')

arg = parser.parse_args()

dataset = arg.datasets
if int(arg.test) == 1:
    data_folder='test'
else:
    data_folder='val'
print(data_folder)
label = open('./local/data/' + dataset + '/'+data_folder+'_label.pkl', 'rb')
label = np.array(pickle.load(label))

rd=[]
rg=[]
data_name=['joint','boneL']
for dn in data_name:
    ri= open('./local/work_dir/' + dataset + '/agcn_'+dn+'_'+data_folder+'/epoch1_'+data_folder+'_score.pkl', 'rb')
    ri = list(pickle.load(ri).items())
    rd.append(ri)
for dn in data_name:
    ri= open('./global/work_dir/' + dataset + '/agcn_'+dn+'_'+data_folder+'/epoch1_'+data_folder+'_score.pkl', 'rb')
    ri = list(pickle.load(ri).items())
    rd.append(ri)
    rg.append(ri)
#,.0,.1,.2,.0,.5]
#alpha = [.2]*5
# alpha=[.3,.4,0.1,0.1,0.1]
# alpha = [0.65685934, 0.19284609, 0.52049417 ,0.06320185 ,0.18706431]
# r1 = open('./work_dir_RUN1/' + dataset + '/agcn_'+data_folder+'_joint/epoch1_'+data_folder+'_score.pkl', 'rb')
# r1 = list(pickle.load(r1).items())
# r2 = open('./work_dir_RUN1/' + dataset + '/agcn_'+data_folder+'_bone/epoch1_'+data_folder+'_score.pkl', 'rb')
# r2 = list(pickle.load(r2).items())
st=""

for al_id,alpha in enumerate(alpha_t):
    right_num = total_num = left_num = 0
    loss_unnorm = 0
    loss_norm=0
    preds=[]
    lbls = []
    pred_probs=[]
    for i in range(len(label[0])):
        _, l = label[:, i]
        # import pdb;pdb.set_trace()
        rs = [rk[i][1] for rk in rd][:5]
        rs2 = [rk[i][1] for rk in rg]
        # rs2   = [0 for rk in rg]
        # print(alpha,rs,rs2)
        # _, r11 = r1[i]
        # _, r22 = r2[i]
        # import pdb;pdb.set_trace()
        # r=r11+alpha*r12
        # unnorm_logits = [wt*(val+val2) for wt,val,val2 in zip(alpha,rs,rs2)]
        norm_logits =   [softmax(wt*(val+val2)) for wt,val,val2 in zip(alpha,rs,rs2)]

        # rank_5 = r.argsort()[-5:]
        # right_num_5 += int(int(l) in rank_5)
        # r = np.argmax(np.sum(unnorm_logits,axis=0))
        # if i ==252:
        #     import pdb;pdb.set_trace()
        # print(i)
        # loss_unnorm += cross_entropy(int(l), softmax(np.sum(unnorm_logits,axis=0)) )
        # right_num += int(r == int(l))
        total_num += 1
        # print(loss)
        nr= np.argmax(np.sum(norm_logits,axis=0))
        pred_prob = np.mean(norm_logits,axis=0)
        pred_probs.append(pred_prob)
        # print(pred_prob)
        loss_norm += cross_entropy(int(l), np.mean(norm_logits,axis=0) )
        left_num += int(nr == int(l))
        preds.append(nr)
        lbls.append(int(l))
    acc = right_num / total_num
    norm_acc= left_num / total_num
    # loss_unnorm /= len(label[0])
    loss_norm /=len(label[0])
    # acc5 = right_num_5 / total_num
    # print("unnorm",alpha,acc,loss_unnorm)
    f1 = f1_score(lbls,preds,average='weighted')
    print("norm ",alpha,"acc:",norm_acc,"loss:",loss_norm , "f1:",f1)
    results_list = []
    for compare in ['ovr','ovo']:
        for avg in ['macro','weighted']:
            print(avg)
            print("ROC OVR",roc_auc_score(lbls, pred_probs,multi_class=compare,average=avg))
            # print("ROC OVO",roc_auc_score(lbls, pred_probs,multi_class='ovo',average=avg))
            results_list.append(np.round(roc_auc_score(lbls, pred_probs,multi_class=compare,average=avg),2))
    
    print(avg ,"PR score",average_precision_score(lbls, pred_probs,average=None))
    for avg in ['micro', 'samples', 'weighted', 'macro']:
        print(avg ,"PR score",average_precision_score(lbls, pred_probs,average=avg))
        results_list.append(np.round(average_precision_score(lbls, pred_probs,average=avg),2))
    store_df(data_folder,results_list,arg.run)  

    # fpr, tpr, thresholds = metrics.roc_curve(lbls, pred_probs, pos_label=0)
    method_list = ['macro ROC OVR','weighted ROC OVR','macro ROC OVO','weighted ROC OVO','micro AUPRC', 'samples AUPRC', 'weighted AUPRC', 'macro AUPRC']

    def plotting():
        y_test=label_binarize(lbls, classes=[*range(n_class)])
        y_score=np.array(pred_probs)
        # precision recall curve
        
        precision = dict()
        recall = dict()
        for i in range(n_class):
            precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                                y_score[:, i])
            plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))
            
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.legend(loc="best")
        plt.title("precision vs. recall curve")
        plt.show()
        # print(fpr, tpr, thresholds)

        fpr = dict()
        tpr = dict()

        for i in range(n_class):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i],
                                        y_score[:, i])
            plt.plot(fpr[i], tpr[i], lw=2, label='class {}'.format(i))

        plt.xlabel("false positive rate")
        plt.ylabel("true positive rate")
        plt.legend(loc="best")
        plt.title("ROC curve")
        plt.show()

        def plot_PR_APscore(targets, predictions, classes):
            enc = OneHotEncoder(handle_unknown='ignore', sparse_output = False)
            targ_multi = enc.fit_transform(targets.reshape(-1, 1))
            color_index = list(mcolors.TABLEAU_COLORS)
            colors = mcolors.TABLEAU_COLORS
            
            fig, ax = plt.subplots(figsize=(6, 6))
            for n_cl, cl in enumerate(classes):  

                f_pre, f_rec, t = precision_recall_curve(targ_multi[:, cl], predictions[:, cl])
                f_f1 = f1_score(targ_multi[:, cl], predictions[:, cl] >= 0.5) # based on the traditional threshold of 0.5
                pr_auc = auc(f_rec, f_pre)
                ap_score = average_precision_score(targ_multi[:, cl], predictions[:, cl])
                ax.plot(
                    f_rec,
                    f_pre,
                    color = colors[color_index[n_cl]],
                    label = f"Class {cl}, PR AUC = {pr_auc:.4f}, AP score = {ap_score:0.4f}, F1 = {f_f1:0.4f}",
                    lw=2,
                    alpha = 0.3,
                )
                        
                ax.set(
                    xlim=[-0.05, 1.05],
                    ylim=[-0.05, 1.05],
                    xlabel="Recall",
                    ylabel="Precision",
                    title=f"PR curves for {len(classes)} classes",
                )
            ax.axis("square")
            ax.legend(loc="lower right")
            plt.show()
        plot_PR_APscore(np.array(lbls), y_score, [0,1,2])  

    # import pdb;pdb.set_trace()
    st+=str(np.round(loss_norm,2))+"/"+str(np.round(norm_acc,2)) +"/"+str(np.round(f1,2)) +"\n"

    print(confusion_matrix(lbls, preds),"\n",classification_report(lbls, preds, target_names=['cor','inc1','inc2'],labels=[i for i in range(n_class)]))

print(st)

