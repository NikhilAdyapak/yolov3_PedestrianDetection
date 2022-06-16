import csv
'''
row_list = [
    ["SN", "Name", "Quotes"],
    [1, "Buddha", "What we think we become"],
    [2, "Mark Twain", "Never regret anything that made you smile"],
    [3, "Oscar Wilde", "Be yourself everyone else is already taken"]
]
with open('quotes.csv', 'w', newline='') as file:
    writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC,
                        delimiter=';', quotechar='*')
    writer.writerows(row_list)

with open('quotes.csv','a',newline='') as file:
    writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC,
                        delimiter=';', quotechar='*')
    writer.writerows(row_list)
'''
import pandas as pd
df = pd.read_csv('output.csv')
print(df.iloc[2,9])
'''
df.loc[5,'gt_x'] = '156'
df.to_csv('output.csv',index = False)

print(df)
#print(df.dtypes)
df = df.astype({'Image_num':int})
print(df)

df.iloc[1] = [2,2,3,4,'NULL','NULL',5,6]
df.to_csv('output.csv',index = False)
'''