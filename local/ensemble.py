import argparse
import pickle
from scipy.special import softmax
from itertools import permutations,combinations_with_replacement
import numpy as np
from tqdm import tqdm
import random
def softmax(x):
  return np.exp(x)/np.sum(np.exp(x),axis=0)
n_class=6
def cross_entropy(y,y_pre):
  y_=[0]*n_class
  y_[y] = 1
  y=y_
  loss=-np.sum(y*np.log(y_pre))
  return loss/float(y_pre.shape[0])

def convex_itertools(count, total):
    candidates = combinations_with_replacement(range(total + 1), r=count)
    return [c for c in candidates if sum(c) == total]

def get_comb(model_count=5):
    a=[list(permutations(i)) for i in convex_itertools(model_count,10)]
    b=[i for c in a for i in c]
    d= list(set(b))
    d=[[i/10 for i in k] for k in d ]
    print("alpha size ",len(d))
    return d

def find_max(acc,norm_acc,max_acc,max_nacc,max_acc_wt,max_nacc_wt,alpha):
    if acc>max_acc:
        max_acc=acc
        max_acc_wt = alpha
    if norm_acc>max_nacc:
        max_nacc=norm_acc
        max_nacc_wt=alpha
    return max_acc,max_acc_wt,max_nacc,max_nacc_wt
def  find_min_loss(loss_unnorm,loss_norm,min_loss,min_nloss,min_loss_wt,min_nloss_wt,alpha):
    if min_loss>loss_unnorm:
        min_loss=loss_unnorm
        min_loss_wt = alpha
    if min_nloss>loss_norm:
        min_nloss=loss_norm
        min_nloss_wt=alpha
    return min_loss,min_loss_wt,min_nloss,min_nloss_wt
parser = argparse.ArgumentParser()
parser.add_argument('--datasets', default='kinetics', choices={'kinetics', 'ntu/xsub', 'ntu/xview'},
                    help='the work folder for storing results')
parser.add_argument('--alpha', default=1, help='weighted summation')
parser.add_argument('--test', default=0, help='test or val data')

arg = parser.parse_args()

dataset = arg.datasets
print(arg.test)
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
# alpha_all=[[1,1,1,1,1], [1,.1,.1,.1,.1] , [1,.5,.5,.2,.2] , [1,.8,.6,.4,.2] , [.5,.5,.2,.2,.2]]
alpha_all =get_comb(5)
alpha_all = random.sample(alpha_all, len(alpha_all))


# for t in range(100):
#     alpha_all.append(np.random.rand(5))
# r1 = open('./work_dir/' + dataset + '/agcn_'+data_folder+'_joint/epoch1_'+data_folder+'_score.pkl', 'rb')
# r1 = list(pickle.load(r1).items())
# r2 = open('./work_dir/' + dataset + '/agcn_'+data_folder+'_bone/epoch1_'+data_folder+'_score.pkl', 'rb')
# r2 = list(pickle.load(r2).items())

max_acc=-1
max_acc_wt=None
max_nacc=-1
max_nacc_wt=None
min_loss=100
min_nloss =100
min_loss_wt=None
min_nloss_wt=None

for alpha in alpha_all:
    right_num = total_num = left_num = 0
    loss_unnorm = 0
    loss_norm = 0
    for i in range(len(label[0])):
        _, l = label[:, i]

        rs = [rk[i][1] for rk in rd]
        # _, r11 = r1[i]
        # _, r22 = r2[i]
        # import pdb;pdb.set_trace()
        # r=r11+alpha*r12
        unnorm_logits = [wt*val for wt,val in zip(alpha,rs)]
        # import pdb;pdb.set_trace()
        norm_logits =   [softmax(wt*val) for wt,val in zip(alpha,rs)]

        # rank_5 = r.argsort()[-5:]
        # right_num_5 += int(int(l) in rank_5)
        r = np.argmax(np.sum(unnorm_logits,axis=0))

        loss_unnorm += cross_entropy(int(l), softmax(np.sum(unnorm_logits,axis=0)) )
        loss_norm += cross_entropy(int(l), np.mean(norm_logits,axis=0) )

        right_num += int(r == int(l))
        total_num += 1

        nr= np.argmax(np.sum(norm_logits,axis=0))
        left_num += int(nr == int(l))

    acc = right_num / total_num
    norm_acc= left_num / total_num

    loss_unnorm = loss_unnorm/len(label[0])
    loss_norm = loss_norm / len(label[0])

    max_acc,max_acc_wt,max_nacc,max_nacc_wt = find_max(acc,norm_acc,max_acc,max_nacc,max_acc_wt,max_nacc_wt,alpha)
    min_loss,min_loss_wt,min_nloss,min_nloss_wt = find_min_loss(loss_unnorm,loss_norm,min_loss,min_nloss,min_loss_wt,min_nloss_wt,alpha)

    # acc5 = right_num_5 / total_num
    print(acc,loss_unnorm,norm_acc,loss_norm,alpha)
    # import pdb;pdb.set_trace()


print("AccNorm",max_acc,max_acc_wt)
print("Acc Unnorm",max_nacc,max_nacc_wt) 

print("loss unnom",min_loss,min_loss_wt)
print("loss norm",min_nloss,min_nloss_wt) 

print(max_acc_wt,",",max_nacc_wt,",",min_loss_wt,",",min_nloss_wt)

alpha_new=[]
alpha_new.append(max_acc_wt) 
alpha_new.append(max_nacc_wt) 
alpha_new.append(min_loss_wt) 
alpha_new.append(min_nloss_wt) 
alpha_new.append([.2,.2,.2,.2,.2]) 

np.save("alpha_t.npy",alpha_new)