import argparse
import pickle
import itertools
from scipy.special import softmax
import numpy as np
from tqdm import tqdm
import os.path
from sklearn.metrics import f1_score

debug  = 0
import pandas as pd
n_class=6
if os.path.isfile('alpha_t.npy') :
    alpha_t = np.load('alpha_t.npy')
    print(alpha_t)
    alpha_t = alpha_t.tolist()
else:
    alpha_t=[[0.2, 0.2, 0.2, 0.2, 0.2]]

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
parser.add_argument('--run', default='RUN5', help='run env')

arg = parser.parse_args()
dir =arg.run

dataset = arg.datasets
if int(arg.test) == 1:
    data_folder='test'
else:
    data_folder='val'

data_sets=[['joint','velocity','acceleration','boneL','boneA'],['joint','velocity','acceleration','boneL'],['joint','velocity','acceleration'],['joint','velocity'],['joint'],['velocity','acceleration','boneL','boneA'],['acceleration','boneL','boneA']]
data_sets=[['joint','velocity','acceleration','boneL','boneA'],['joint','velocity','acceleration','boneL'],['joint','velocity','acceleration','boneA'],['joint','velocity','boneL','boneA'],['joint','acceleration','boneL','boneA'],['velocity','acceleration','boneL','boneA']]
data_sets=[['joint','velocity','acceleration','boneL','boneA'],['joint','velocity','acceleration','boneL'],['joint','velocity','acceleration'],['joint','velocity','boneL'],['joint','acceleration','boneL'],['velocity','acceleration','boneL']]
data_sets=[['joint','velocity','acceleration','boneL','boneA'],['velocity','acceleration','boneL'],['velocity','acceleration'],['velocity','boneL'],['acceleration','boneL'],['boneA','boneL']]
data_sets=[['joint'],['velocity'],['acceleration'],['boneL'],['boneA'],['joint','velocity','acceleration','boneL','boneA']]
data_sets=[['velocity','acceleration','boneL','boneA'],['joint','acceleration','boneL','boneA'],['joint','velocity','boneL','boneA'],['joint','velocity','acceleration','boneA'],['joint','velocity','acceleration','boneL']]

df = pd.DataFrame()
df["RUN"] = [dir]
if debug:
    print(data_sets)
data_list = ['joint','velocity','acceleration','boneL','boneA']
for ele in range(1,6):
    df = pd.DataFrame()
    df["RUN"] = [dir]
    combinations = list(itertools.combinations(data_list, ele))
    if debug:
        print(ele)
    data_folder='val'

    for data_name in combinations: 
        df[str(data_name)+"_val"] = np.nan # create empty columns
        

        # print(data_folder)
        label = open('./local/data/' + dataset + '/'+data_folder+'_label.pkl', 'rb')
        label = np.array(pickle.load(label))


        rd=[]
        rg=[]   
        # data_name=['joint','velocity','acceleration','boneL','boneA']
        if debug:
            print(data_name)
        for dn in data_name:
            ri= open('./local/work_dir_'+dir+'/' + dataset + '/agcn_'+dn+'_'+data_folder+'/epoch1_'+data_folder+'_score.pkl', 'rb')
            ri = list(pickle.load(ri).items())
            rd.append(ri)
        for dn in data_name:
            ri= open('./global/work_dir_'+dir+'/' + dataset + '/agcn_'+dn+'_'+data_folder+'/epoch1_'+data_folder+'_score.pkl', 'rb')
            ri = list(pickle.load(ri).items())
            rd.append(ri)
            rg.append(ri)
        #,.0,.1,.2,.0,.5]
        #alpha = [.2]*5
        # alpha=[.3,.4,0.1,0.1,0.1]
        # alpha = [0.65685934, 0.19284609, 0.52049417 ,0.06320185 ,0.18706431]
        # r1 = open('./work_dir_RUN5_RUN1/' + dataset + '/agcn_'+data_folder+'_joint/epoch1_'+data_folder+'_score.pkl', 'rb')
        # r1 = list(pickle.load(r1).items())
        # r2 = open('./work_dir_RUN5_RUN1/' + dataset + '/agcn_'+data_folder+'_bone/epoch1_'+data_folder+'_score.pkl', 'rb')
        # r2 = list(pickle.load(r2).items())
        st=""

        for al_id,alpha in enumerate(alpha_t):
            right_num = total_num = left_num = 0
            loss_unnorm = 0
            loss_norm=0
            preds=[]
            lbls = []
            for i in range(len(rd[0])):
                _, l = label[:, i]

                rs = [rk[i][1] for rk in rd][:5]
                rs2 = [rk[i][1] for rk in rg]
                # rs2   = [0 for rk in rg]

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
                loss_norm += cross_entropy(int(l), np.mean(norm_logits,axis=0) )
                left_num += int(nr == int(l))
                preds.append(nr)
                lbls.append(int(l))
            acc = right_num / total_num
            norm_acc= left_num / total_num
            # loss_unnorm /= len(rd[0])
            loss_norm /=len(rd[0])
            # acc5 = right_num_5 / total_num
            # print("unnorm",alpha,acc,loss_unnorm)
            f1 = f1_score(lbls,preds,average='weighted')
            # print("acc:",norm_acc,"loss:",loss_norm , "f1:",f1)

            st+=str(np.round(loss_norm,2))+"/"+str(np.round(norm_acc,2))+"/"+str(np.round(f1,2)) 
            df.loc[0,str(data_name)+"_val"] = [str(np.round(f1,2))]
        if debug:
            print(st)



    # data_folder='val'
    data_folder='test'

    for data_name in combinations: 

        df[str(data_name)+"_test"] = np.nan # create empty columns
        # print(data_folder)
        label = open('./local/data/' + dataset + '/'+data_folder+'_label.pkl', 'rb')
        label = np.array(pickle.load(label))


        rd=[]
        rg=[]   
        # data_name=['joint','velocity','acceleration','boneL','boneA']
        # print(data_name)
        for dn in data_name:
            ri= open('./local/work_dir_'+dir+'/' + dataset + '/agcn_'+dn+'_'+data_folder+'/epoch1_'+data_folder+'_score.pkl', 'rb')
            ri = list(pickle.load(ri).items())
            rd.append(ri)
        for dn in data_name:
            ri= open('./global/work_dir_'+dir+'/' + dataset + '/agcn_'+dn+'_'+data_folder+'/epoch1_'+data_folder+'_score.pkl', 'rb')
            ri = list(pickle.load(ri).items())
            rd.append(ri)
            rg.append(ri)
        #,.0,.1,.2,.0,.5]
        #alpha = [.2]*5
        # alpha=[.3,.4,0.1,0.1,0.1]
        # alpha = [0.65685934, 0.19284609, 0.52049417 ,0.06320185 ,0.18706431]
        # r1 = open('./work_dir_RUN5_RUN1/' + dataset + '/agcn_'+data_folder+'_joint/epoch1_'+data_folder+'_score.pkl', 'rb')
        # r1 = list(pickle.load(r1).items())
        # r2 = open('./work_dir_RUN5_RUN1/' + dataset + '/agcn_'+data_folder+'_bone/epoch1_'+data_folder+'_score.pkl', 'rb')
        # r2 = list(pickle.load(r2).items())
        st=""

        for al_id,alpha in enumerate(alpha_t):
            right_num = total_num = left_num = 0
            loss_unnorm = 0
            loss_norm=0
            preds=[]
            lbls = []
            for i in range(len(rd[0])):
                _, l = label[:, i]

                rs = [rk[i][1] for rk in rd][:5]
                rs2 = [rk[i][1] for rk in rg]
                # rs2   = [0 for rk in rg]

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
                loss_norm += cross_entropy(int(l), np.mean(norm_logits,axis=0) )
                left_num += int(nr == int(l))
                preds.append(nr)
                lbls.append(int(l))
            acc = right_num / total_num
            norm_acc= left_num / total_num
            # loss_unnorm /= len(rd[0])
            loss_norm /=len(rd[0])
            # acc5 = right_num_5 / total_num
            # print("unnorm",alpha,acc,loss_unnorm)
            f1 = f1_score(lbls,preds,average='weighted')
            # print("acc:",norm_acc,"loss:",loss_norm , "f1:",f1)

            st+=str(np.round(loss_norm,2))+"/"+str(np.round(norm_acc,2))+"/"+str(np.round(f1,2)) + "\n"
            df.loc[0,str(data_name)+"_test"] = [str(np.round(f1,2))]
        if debug :
            print(st)
# import pdb;pdb.set_trace()
    if os.path.isfile(str(ele)+'ensemble_ablation.csv'):
        df_full = pd.read_csv(str(ele)+'ensemble_ablation.csv')

        df = pd.concat([df_full, df], axis=0)
    df.to_csv(str(ele)+'ensemble_ablation.csv',index=False)
