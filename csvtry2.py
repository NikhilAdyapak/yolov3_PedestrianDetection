import pandas as pd

df = pd.read_csv('output.csv')
dict = {'Sno':2,'Image_num':2,'x':3,'y':4,'w':5,'h':10}
df = df.append(dict,ignore_index = True)
print(df)
df.loc[1 , 'gt_h'] = int(12)
df.to_csv('output.csv',index = False)
print(df) 