import pandas as pd
import numpy as np
df_full = pd.read_csv('compare_AUC_2sAGCNFeat.csv',index_col='RUNS')
mean_row = len(df_full)+2
std_row=len(df_full)+3
df_full.loc[mean_row,:]=df_full.mean(axis=0)
df_full.loc[std_row,:]=np.round(df_full.std(axis=0),5)
# df_full.loc[mean_row,"RUNS"]="Mean"
# df_full.loc[std_row,"RUNS"]="STD"
# org_method_list = ['weighted ROC OVO','weighted AUPRC','macro ROC OVR','weighted ROC OVR','macro ROC OVO','micro AUPRC', 'samples AUPRC',  'macro AUPRC']
new_method_list = ['weighted ROC OVO','weighted AUPRC','macro ROC OVR','weighted ROC OVR','macro ROC OVO','micro AUPRC', 'samples AUPRC',  'macro AUPRC']
col=[]
for c in new_method_list:
    for id in ['val','test']:
        col.append(c+'_'+id)
# print(df_full.columns)
# print(col)
df_full = df_full.loc[:,col]
df_full.to_csv('compare_AUC_2sAGCNFeat_mean.csv',index=True)

