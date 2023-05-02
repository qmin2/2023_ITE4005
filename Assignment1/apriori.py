import pandas as pd
import os
from collections import Counter
import argparse
from itertools import combinations

def self_join(itemsets):
    '''
    Parameters:
    itemsets (set of frozensets): The set of itemsets to self-join.
    '''
    joined_itemsets = set()
    for itemset1 in itemsets:
        for itemset2 in itemsets:
            if itemset1 != itemset2:
                joined_itemset = frozenset(itemset1.union(itemset2))
                if len(joined_itemset) == len(itemset1) + 1:
                    joined_itemsets.add(joined_itemset)
    return joined_itemsets


def generate_association_rules(frequent_itemsets, min_confidence, total_num_of_db):
    """
    Generates all association rules that satisfy the minimum confidence threshold from a set of frequent itemsets.
    
    Parameters:
    frequent_itemsets (dict): A dictionary of frequent itemsets, where the keys are the itemsets and the values are their corresponding support counts.
    min_confidence (float): The minimum confidence threshold for association rules.
    
    Returns:
    list of tuple: A list of tuples, where each tuple represents an association rule in the form (antecedent, consequent, confidence).
    """
    association_rules = []
    for itemset in frequent_itemsets: # keys 안해도 될듯
        for i in range(1, len(itemset)):
            antecedents = combinations(itemset, i)
            for antecedent in antecedents:
                antecedent = frozenset(antecedent)
                consequent = itemset.difference(antecedent) # 차집합
                if len(consequent) > 0:
                    if frequent_itemsets[antecedent] > 0:
                        confidence = frequent_itemsets[itemset] / frequent_itemsets[antecedent]
                    else:
                        continue
                    if confidence >= min_confidence:
                        association_rules.append((antecedent, consequent, frequent_itemsets[itemset]/total_num_of_db*100, confidence*100))
    return association_rules


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--minimum_support', type=int, help = 'this integer stands for the percentage, not the value itself')
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)

    args = parser.parse_args()


    file_path = os.path.join(args.input_file)
    df_db = pd.DataFrame() # blank DataFrame to store results
    f = open(file_path)
    database = []
    for line in f.readlines():
        line = line.strip()

        num_list = [int(num) for num in line.split("\t")]
        database.append(num_list)
        df_db = pd.DataFrame({'transaction': database})

    init = set()
    for row in df_db.itertuples():
        init = set.union(init, row[1])
    init = sorted(init)  # this is the total element


    total_num_of_db = len(df_db)
    min_sup = args.minimum_support/100
    sup_freq = int(min_sup * total_num_of_db)
    # print(sup_freq) # 25

    c = Counter()
    for i in init:
        for row in df_db.itertuples():
            if (i in row[1]):
                c[i]+=1
    
    l = Counter()
    for i in c:
        if(c[i] >= sup_freq):
            l[frozenset([i])]+=c[i]
    total_frequncy = l # L1

    while True:
        temp = list(l)
        next_candi = self_join(temp)

        c = Counter()
        for i in next_candi:
            c[i] = 0
            for row in df_db.itertuples():
                temp = set(row[1])
                if(i.issubset(temp)): # prunning
                    c[i]+=1
        
        l = Counter()
        for i in c:
            if(c[i] >= sup_freq):
                l[i]+=c[i]
        if(len(l) == 0):
            break
        total_frequncy += l

    group = generate_association_rules(total_frequncy, 0, total_num_of_db)
    # print(len(total_frequncy)) # 303
    # print(len(group)) # 1066


    with open(args.output_file, 'w') as f:
        for tpl in group:
            fset1 = '{' + ','.join(map(str, tpl[0])) + '}'
            fset2 = '{' + ','.join(map(str, tpl[1])) + '}'
            sup = tpl[2]
            conf = tpl[3]
            f.write('{}\t{}\t{:.2f}\t{:.2f}\n'.format(fset1, fset2, sup, conf))


if __name__ == '__main__':
    main()