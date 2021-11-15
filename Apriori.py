#!/usr/bin/env python
# coding: utf-8

# # Apriori algorithm 
# #### Apriori algorithm refers to the algorithm which is used to calculate the association rules between objects. It means how two or more objects are related to one another. In other words, we can say that the apriori algorithm is an association rule leaning that analyzes that people who bought product A also bought product B.
# 
# #### The primary objective of the apriori algorithm is to create the association rule between different objects. The association rule describes how two or more objects are related to one another. Apriori algorithm is also called frequent pattern mining. 
# 
# ## Support
# #### Support refers to the default popularity of any product. You find the support as a quotient of the division of the number of transactions comprising that product by the total number of transactions. 
# 
# ## Confidence
# #### Confidence refers to the possibility that the customers bought both biscuits and chocolates together. So, you need to divide the number of transactions that comprise both biscuits and chocolates by the total number of transactions to get the confidence.

# In[1]:


#importing all the neccessary libraries
import pandas as pd
import numpy as np
from itertools import combinations


# In[20]:


def read_file_and_prepare_data(filename="BestBuy.csv",colname = "Transaction"):
    """
    Reading the csv file to get the transactions/items. Once all the transactions are read, 
    this functions finds the unique values of the entire transaction and assigns each unique item
    to a unique value such as 0,1,2 and so on. All the transactions are converted to a list of numbers.
    By default the file is BestBuy.csv
    """

    read_df = pd.read_csv(filename)
    print(read_df.columns)
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


# In[21]:


def convert_numbers_to_names(ck):
    """
    In the read_file_and_prepare_data function we assigned each unique item to a unique number.
    This fuction is used to revert the numbers back to their names (into understandable form instead
    of numbers) while printing it back to the user.
    """
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


# In[22]:


def create_1_itemset(X):
    """
    create the 1-item candidate,
    it's basically creating a frozenset for each unique item
    and storing them in a list
    """
    c1 = []
    for transaction in X:
        for t in transaction:
            t = frozenset([t])
            if t not in c1:
                c1.append(t)
    return c1


# In[23]:


def create_k_itemset(freq_item, k):
    """create the list of k-item(2,3,4...)items"""
    ck = []
    
    # for generating candidate of size two (2-itemset)
    if k == 0:
        for f1, f2 in combinations(freq_item, 2):
            item = f1 | f2 # union of two sets
            ck.append(item)
    else:    
        for f1, f2 in combinations(freq_item, 2):       
            # if the two (k+1)-item sets has
            # k common elements then they will be
            # unioned to be the (k+2)-item candidate
            intersection = f1 & f2
            if len(intersection) == k:
                item = f1 | f2
                if item not in ck:
                    ck.append(item)
    return ck


# In[24]:


def create_freq_item(X, ck, min_support):
    """
    filters the candidate with the specified
    minimum support
    """
    # loop through the transaction and compute
    # the count for each candidate (item)
    item_count = {}
    for transaction in X:
        for item in ck:
            if item.issubset(transaction):
                if item not in item_count: 
                    item_count[item] = 1
                else: 
                    item_count[item] += 1    
    
    n_row = X.shape[0]
    freq_item = []
    item_support = {}
    
    # if the support of an item is greater than the 
    # min_support, then it is considered as frequent
    for item in item_count:
        support = item_count[item] / n_row
        if support >= min_support:
            freq_item.append(item)
        
        item_support[item] = support
        
    return freq_item, item_support


# In[32]:


def apriori(X, min_support=0.5):
    """
    pass in the transaction data and the minimum support 
    threshold to obtain the frequent itemset. Also
    store the support for each itemset, they will
    be used in the rule generation step
    """

    # the candidate sets for the 1-item is different,
    # create them independently from others
    c1 = create_1_itemset(X)
    freq_item, item_support_dict = create_freq_item(X, c1, min_support)
    freq_items = [freq_item]
    freq=[]
    k = 0
    while len(freq_items[k]) > 0:
        freq_item = freq_items[k]
        ck = create_k_itemset(freq_item, k)       
        freq_item, item_support = create_freq_item(X, ck, min_support)
        freq_items.append(freq_item)
        item_support_dict.update(item_support)
        k += 1
    freq_print = convert_numbers_to_names(freq_items)
    return freq_print,freq_items, item_support_dict


# In[26]:


def create_rules(freq_items, item_support_dict, min_confidence):
    """
    create the association rules, the rules will be a list.
    each element is a tuple of size 4, containing rules'
    left hand side, right hand side, confidence and lift
    """
    association_rules = []

    # for the list that stores the frequent items, loop through
    # the second element to the one before the last to generate the rules
    # because the last one will be an empty list. It's the stopping criteria
    # for the frequent itemset generating process and the first one are all
    # single element frequent itemset, which can't perform the set
    # operation X -> Y - X
    for idx, freq_item in enumerate(freq_items[1:(len(freq_items) - 1)]):
        for freq_set in freq_item:
            
            # start with creating rules for single item on
            # the right hand side
            subsets = [frozenset([item]) for item in freq_set]
            rules, right_hand_side = compute_conf(freq_items, item_support_dict, 
                                                  freq_set, subsets, min_confidence)
            association_rules.extend(rules)
            
            # starting from 3-itemset, loop through each length item
            # to create the rules, as for the while loop condition,
            # e.g. suppose you start with a 3-itemset {2, 3, 5} then the 
            # while loop condition will stop when the right hand side's
            # item is of length 2, e.g. [ {2, 3}, {3, 5} ], since this
            # will be merged into 3 itemset, making the left hand side
            # null when computing the confidence
            if idx != 0:
                k = 0
                while len(right_hand_side[0]) < len(freq_set) - 1:
                    ck = create_k_itemset(right_hand_side, k = k)
                    rules, right_hand_side = compute_conf(freq_items, item_support_dict,
                                                          freq_set, ck, min_confidence)
                    association_rules.extend(rules)
                    k += 1   
    rules_df = pd.DataFrame(association_rules,columns=['First_item',"buys","Second_item","confidence_value"])
    list_first_item = [list(rules_df['First_item'])]
    list_second_item = [list(rules_df['Second_item'])]
    rules_df["First_item"] = convert_numbers_to_names(list_first_item)
    rules_df["Second_item"] = convert_numbers_to_names(list_second_item)
    return rules_df


# In[27]:


def compute_conf(freq_items, item_support_dict, freq_set, subsets, min_confidence=0.5):
    """
    create the rules and returns the rules info and the rules's
    right hand side (used for generating the next round of rules) 
    if it surpasses the minimum confidence threshold
    """
    rules = []
    right_hand_side = []
    
    for rhs in subsets:
        # create the left hand side of the rule
        # and add the rules if it's greater than
        # the confidence threshold
        lhs = freq_set - rhs
        conf = item_support_dict[freq_set] / item_support_dict[lhs]
        if conf >= min_confidence:
            rules_info = lhs,"->", rhs, conf
            rules.append(rules_info)
            right_hand_side.append(rhs)
            
    return rules, right_hand_side


# In[36]:


def run():
    """
    In this function, all the required inputs are provided by the user 
    such as the filename, support value and confidence value.
    These inputs are provided in the function call
    """
    file = input("Enter the File Name(CSV) [if in different directory please provide the address with the file]\n")
    if ".xlsx" in file:
        read_file = pd.read_excel(file)
        read_file.to_csv (file, index = None, header=True)
    X = read_file_and_prepare_data(file)
    try:
        min_support = float(input("Enter the SUPPORT value : "))
        if not min_support >=0.0 and min_support <=1.0:
            min_support = 0.5
    except:
        min_support = 0.5
    try:
        min_confidence = float(input("Enter the CONFIDENCE value : "))
        if not min_confidence >= 0.0 and min_confidence <= 1.0:
            min_confidence = 0.5
    except:
        min_confidence = 0.5
    freq_print,freq_items, item_support_dict = apriori(X, min_support)
    print("-"*120)
    print("The Minimum support value is : ",min_support)
    print("The Minimum Confidence value is : ",min_confidence)
    print("-"*120)
    print("The Frequent itemset which is above or equal to the given support value: \n")
    print(freq_print)
    rules_df = create_rules(freq_items, item_support_dict, min_confidence)
    print("-"*120)
    print("The rules with their confidence values are : \n")
    print(rules_df)
    rules_df.to_csv("output.csv")
    print("-"*120)
    print("The rules with their confidence value are available in output.csv file")
    print("-"*120)
    print("-"*120)


# In[38]:


run()


# # The csv file needs to be formatted before running the apriori algorithm
# ### The csv file looks
# ![excel](excel.png)
# ### This file is formatted as shown in the picture below
# ![formatted](formatted.png)

# In[22]:


import pandas as pd
def file_format(filename):
    """This Function is used to reformat the file"""
    df = pd.read_csv(filename)
    list_values = df.values.tolist()
    split=[]
    for i in list_values:
        for j in i:
            split += [j.split(" ", maxsplit =1)]
    return_df = pd.DataFrame(split, columns=['Transaction ID', 'Transaction'])
    return_df.to_csv(filename,index=False)
    print(return_df)
#file_format("custom_dataset.csv")


# In[ ]:




