#coding=utf-8
'''
@author: DMao
@time: 28.05.20    14:38
'''
import pandas as pd
import numpy as np
print(pd.__version__)
'''
data = pd.Series([1, 2, 3, 4])
print(data)
dic_data = {'key1': 123, 'key2': 456, 'key3': 789}
d2 = pd.Series(dic_data)

dic_data2 = {'key1': 'value1', 'key2':'value2', 'key3': 'value3'}
d3 = pd.Series(dic_data2)

df = pd.DataFrame({'column1':d2, 'column2':d3})
print('-----', df['column2'])

A = pd.DataFrame(np.random.randint(0,20,(2,2)))
print(A)
x = A.stack().mean()
print(x)

A = np.random.randint(10, size=(3,4))
print(A)
df = pd.DataFrame(A, columns=list('QRST'))
print(df)
print('------')
hf = df.iloc[0, ::2]
print(hf)
'''

index = [ ('abc', 2000), ('cde', 2010), ('abc', 2010), ('cde', 2000)]
index = pd.MultiIndex.from_tuples(index)
print(index)

part1 = pd.Series({'a': 123, 'b': 234, 'c': 345, 'd': 456})
part2 = pd.Series({'a': 'aaa', 'b': 'bbb', 'c': 'ccc', 'd': 'ccc'})
df = pd.DataFrame({'state': part1, 'population': part2, 'column3': part1})
print(df)
print(df['state'][df['population']!='bbb'].unique())
#print(df[['state','column3']])
