import argparse
import pickle
from scipy.special import softmax
import numpy as np
from tqdm import tqdm
import os
if os.path.isfile('alpha_t.npy') :
    alpha_t = np.load('alpha_t.npy')
    print(alpha_t)
    alpha_t = alpha_t.tolist()
else:
    alpha_t=[[0.2, 0.1, 0.0, 0.2, 0.5] , [0.3, 0.1, 0.0, 0.2, 0.4] , [0.2, 0.2, 0.0, 0.5, 0.1] , [0.2, 0.2, 0.0, 0.3, 0.3], [0.2, 0.2, 0.2, 0.2, 0.2]]
# alpha_t=[[0.1, 0.0, 0.4, 0.4, 0.1], [0.0, 0.0, 0.4, 0.4, 0.2],[0.1, 0.3, 0.0, 0.2, 0.4],[0.2, 0.2, 0.0, 0.3, 0.3]]
n_class=6
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

arg = parser.parse_args()

dataset = arg.datasets
if int(arg.test) == 1:
    data_folder='test'
else:
    data_folder='val'
print(data_folder)
label = open('./data/' + dataset + '/'+data_folder+'_label.pkl', 'rb')
label = np.array(pickle.load(label))

rd=[]
data_name=['joint','velocity','acceleration','boneL','boneA']
for dn in data_name:
    ri= open('./work_dir/' + dataset + '/agcn_'+dn+'_'+data_folder+'/epoch1_'+data_folder+'_score.pkl', 'rb')
    ri = list(pickle.load(ri).items())
    rd.append(ri)
# alpha=[.1,.0,.4,.4,.1]
#alpha = [1.0]*5
# alpha=[.3,.4,0.1,0.1,0.1]
# alpha = [0.65685934, 0.19284609, 0.52049417 ,0.06320185 ,0.18706431]
# r1 = open('./work_dir/' + dataset + '/agcn_'+data_folder+'_joint/epoch1_'+data_folder+'_score.pkl', 'rb')
# r1 = list(pickle.load(r1).items())
# r2 = open('./work_dir/' + dataset + '/agcn_'+data_folder+'_bone/epoch1_'+data_folder+'_score.pkl', 'rb')
# r2 = list(pickle.load(r2).items())
st=""
for al_id,alpha in enumerate(alpha_t):
    print(alpha)
    right_num = total_num = left_num = 0
    loss_unnorm = 0
    loss_norm=0
    for i in range(len(label[0])):
        _, l = label[:, i]

        rs = [rk[i][1] for rk in rd]
        # _, r11 = r1[i]
        # _, r22 = r2[i]
        # import pdb;pdb.set_trace()
        # r=r11+alpha*r12
        unnorm_logits = [wt*val for wt,val in zip(alpha,rs)]
        norm_logits =   [softmax(wt*val) for wt,val in zip(alpha,rs)]

        # rank_5 = r.argsort()[-5:]
        # right_num_5 += int(int(l) in rank_5)
        r = np.argmax(np.sum(unnorm_logits,axis=0))
        # if i ==252:
        #     import pdb;pdb.set_trace()
        # print(i)
        loss_unnorm += cross_entropy(int(l), softmax(np.sum(unnorm_logits,axis=0)) )
        right_num += int(r == int(l))
        total_num += 1
        # print(loss)
        nr= np.argmax(np.sum(norm_logits,axis=0))
        loss_norm += cross_entropy(int(l), np.mean(norm_logits,axis=0) )
        left_num += int(nr == int(l))

    acc = right_num / total_num
    norm_acc= left_num / total_num
    loss_unnorm /= len(label[0])
    loss_norm /=len(label[0])
    # acc5 = right_num_5 / total_num
    print("unnorm",alpha,acc,loss_unnorm)
    print("norm ",alpha,norm_acc,loss_norm)
    
    if al_id==0:
        st+=str(np.round(loss_unnorm,2))+"/"+str(np.round(acc,2)) +"\n"
    elif al_id==1:
        st+=str(np.round(loss_norm,2))+"/"+str(np.round(norm_acc,2)) +"\n"
    elif al_id==2:
        st+=str(np.round(loss_unnorm,2))+"/"+str(np.round(acc,2)) +"\n"
    elif al_id==3:
        st+=str(np.round(loss_norm,2))+"/"+str(np.round(norm_acc,2)) +"\n"
    else:
        st+=str(np.round(loss_unnorm,2))+"/"+str(np.round(acc,2)) +"\n"
        st+=str(np.round(loss_norm,2))+"/"+str(np.round(norm_acc,2)) +"\n"
print(st)
