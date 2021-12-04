#!/usr/bin/env python
# coding: utf-8

# # Apriori

# In[1]:


#importing all the neccessary libraries
import pandas as pd
import numpy as np
from itertools import combinations


# In[2]:


def read_file_and_prepare_data(filename="BestBuy.csv",colname = "Transaction"):

    read_df = pd.read_csv(filename)
    lst=[]
    for i in read_df[colname]:
        lst += [i.split(',')]
    unique=[]
    list1=[]
    for a in lst:
        l1=[]
        for b in a:
            b= b.lstrip().rstrip()
            l1+=[b]
            if b not in unique:
                unique+=[b]
        l1.sort()
        list1+=[l1]
        unique.sort()
        list2=[]
    global dict1, key_list,val_list, X
    dict1 = { unique[i] : i for i in range(0, len(unique) ) }
    key_list = list(dict1.keys())
    val_list = list(dict1.values())

    for a in list1:
        l1 = []
        for b in a:
            l1 += [dict1[b]]
        list2 += [l1]
    return np.array(list2)


# In[3]:


def freq_itemset(X, itemset, min_support):

    items = {}
    for a in X:
        for item in itemset:
            if item.issubset(a):
                if item not in items: 
                    items[item] = 1
                else: 
                    items[item] += 1    
    
    rows = X.shape[0]
    frequentset = []
    item_support = {}
    for item in items:
        s = items[item] / rows
        if s >= min_support:
            frequentset.append(item)
        
        item_support[item] = s
        
    return frequentset, item_support


# In[4]:


def support(X, min_support=0.5):

    itemset1 = []
    for i in X:
        for j in i:
            j = frozenset([j])
            if j not in itemset1:
                itemset1.append(j)
    
    fi, item_support_dict = freq_itemset(X, itemset1, min_support)
    fitems = [fi]
    freq=[]
    k = 0
    while len(fitems[k]) > 0:
        fi = fitems[k]
        itemset = []
        if k == 0:
            for a, b in combinations(fi, 2):
                item = a | b
                itemset.append(item)
        else:    
            for a, b in combinations(fi, 2):       
                intersection = a & b
                if len(intersection) == k:
                    item = a | b
                    if item not in itemset:
                        itemset.append(item)     
       
        fi, supportitems = freq_itemset(X, itemset, min_support)
        fitems.append(fi)
        item_support_dict.update(supportitems)
        k += 1
    freq_print = convert_numbers_to_names(fitems)
    return freq_print,fitems, item_support_dict


# In[5]:


def confidence(fitems, item_support_dict, min_confidence):

    association_rules = []
    for index, freq in enumerate(fitems[1:(len(fitems) - 1)]):
        for freq_set in freq:
            subsets = [frozenset([item]) for item in freq_set]
            rules, denominator = conf( item_support_dict, 
                                                  freq_set, subsets, min_confidence)
            association_rules.extend(rules)

            if index != 0:
                k = 0
                while len(denominator[0]) < len(freq_set) - 1:
                    itemset = []
                    if k == 0:
                        for a, b in combinations(denominator, 2):
                            item = a | b
                            itemset.append(item)
                    else:    
                        for a, b in combinations(denominator, 2):       
                            intersection = a & b
                            if len(intersection) == k:
                                item = a | b
                                if item not in itemset:
                                    itemset.append(item)     
                    
                    
                    rules, denominator = conf( item_support_dict,
                                                          freq_set, itemset, min_confidence)
                    association_rules.extend(rules)
                    k += 1   
    rules_df = pd.DataFrame(association_rules,columns=['First_item',"buys","Second_item","confidence_value"])
    list_first_item = [list(rules_df['First_item'])]
    list_second_item = [list(rules_df['Second_item'])]
    rules_df["First_item"] = convert_numbers_to_names(list_first_item)
    rules_df["Second_item"] = convert_numbers_to_names(list_second_item)
    return rules_df


# In[6]:


def conf( item_support_dict, fset, sets, min_confidence=0.5):

    rules = []
    denominator = []
    
    for a in sets:
        b = fset - a
        cv = item_support_dict[fset] / item_support_dict[b]
        if cv >= min_confidence:
            info = b,"->", a, cv
            rules.append(info)
            denominator.append(a)
            
    return rules, denominator


# In[7]:


def convert_numbers_to_names(ck):

    list_ = [list(x) for x in ck]
    len1 = (len(list_))
    list_form=[]
    for i in range(0,len1):
        if len(list_[i]) >= 1:
            list_form1 = list_[i]
            form=[]
            for innerlist in list_form1:
                a=[]
                for i in innerlist:
                    position = val_list.index(i)
                    a += [key_list[position]]
                form += [a]
            list_form +=form
    return list_form


# In[8]:


def run():
    file = input("Enter the File Name(CSV) [if in different directory please provide the address with the file]\n")
    if ".xlsx" in file:
        read_file = pd.read_excel(file)
        read_file.to_csv (file, index = None, header=True)
    X = read_file_and_prepare_data(file)
    while True:
        try:
            min_support = float(input("Enter the SUPPORT value in the range of (0.0 - 1.0) : "))
            if min_support >=0.0 and min_support <=1.0:
                break
        except:
            min_support = 0.5
    while True:
        try:
            min_confidence = float(input("Enter the CONFIDENCE value in the range of (0.0 - 1.0) : "))
            if min_confidence >= 0.0 and min_confidence <= 1.0:
                break
        except:
            min_confidence = 0.5
    freq_print,freq_items, item_support_dict = support (X, min_support)
    print("-"*120)
    print("The Minimum support value is : ",min_support)
    print("The Minimum Confidence value is : ",min_confidence)
    print("-"*120)
    print("The Frequent itemset which is above or equal to the given support value: \n")
    print(freq_print)
    rules_df = confidence(freq_items, item_support_dict, min_confidence)
    print("-"*120)
    print("The rules with their confidence values are : \n")
    print(rules_df)
    rules_df.to_csv("output.csv")
    print("-"*120)
    print("The rules with their confidence value are available in output.csv file")
    print("-"*120)
    print("-"*120)


# In[21]:


run()


# In[ ]:




