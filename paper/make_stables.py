import sys, pandas as pd
D=sys.argv[1]; O=sys.argv[2]
main16=['ZeroGaussian','EWMAGaussian','GARCH-Norm','RollingEmpirical','GaussianTCN','MCDropout','DeepEnsembles','Evidential','Isotonic','SplitConformal','OnlineQuantile','RandomNoise','MixUp','FeaturePGD','AdvReg','SCALE-R']
w=pd.read_csv(D+'metrics_wide.csv'); w=w[w.market!='SP500ETF']
ms=['skill_vs_zero','ECE','PICP90','NMPIW90','NWIS','UA_width','nRMSE']
s1=w[w.method.isin(main16)].groupby(['split','method','asset_class','market'])[ms].mean().reset_index()
s1['period']=s1.split.map({'historical':'2020-2022','recent':'2023-2025'})
s1=s1[['period','method','asset_class','market']+ms].sort_values(['period','method','market'])
s1.to_excel(O+'S1_Table.xlsx',index=False,sheet_name='per_target_seed_mean')
mm=pd.read_csv(D+'macro_mean_sd.csv'); mm.insert(0,'period',mm.split.map({'historical':'2020-2022','recent':'2023-2025'})); mm=mm.drop(columns='split')
mm.to_excel(O+'S2_Table.xlsx',index=False,sheet_name='target_macro_mean_sd')
lo=pd.read_csv(D+'leave_one_asset_class_out.csv'); lo=lo[lo.method.isin(main16)]
s3=lo.groupby(['split','excluded_asset_class','method'])[['NWIS','ECE','PICP90','skill_vs_zero']].agg(['mean','std'])
s3.columns=['_'.join(c) for c in s3.columns]; s3=s3.reset_index(); s3.insert(0,'period',s3.split.map({'historical':'2020-2022','recent':'2023-2025'})); s3=s3.drop(columns='split')
s3.to_excel(O+'S3_Table.xlsx',index=False,sheet_name='leave_one_asset_class_out')
pb=pd.read_csv(D+'paired_block_bootstrap.csv'); pb.insert(0,'period',pb.split.map({'historical':'2020-2022','recent':'2023-2025'})); pb=pb.drop(columns='split')
pb.to_excel(O+'S4_Table.xlsx',index=False,sheet_name='predefined_paired_intervals')
print(len(s1),len(mm),len(s3),len(pb))
