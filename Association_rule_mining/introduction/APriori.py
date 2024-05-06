# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 23:47:01 2024

@author: terzi
"""

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
           ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
           ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]


te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary,columns=te.columns_)

rules = apriori(df,min_support=0.6,use_colnames=True)

print(rules,'\n')

desired_itemset = frozenset({'Eggs', 'Onion'})
y = frozenset({'Kidney Beans'})

#issubset = altküme true veya false döndürür
filtered_rules = rules[rules['itemsets'].apply(lambda x: desired_itemset.issubset(x))]
#lambda parametreler: ifade ; Anonim fonksiyonlar, adı olmayan ve genellikle tek bir satırda yazılan fonksiyonlardır.
# apply : apply, Pandas DataFrame veya Serilerindeki her bir elemana bir fonksiyon uygulamak için kullanılan bir yöntemdir.
#   df.apply(lambda x: x * 2) 
filtered_rules_with_kidney = filtered_rules[filtered_rules['itemsets'].apply(lambda x: y.issubset(x))]

print(filtered_rules_with_kidney[['itemsets']])

