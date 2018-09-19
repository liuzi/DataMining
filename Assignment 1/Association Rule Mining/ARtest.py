# import sys
# import os


def read_csv(filepath):
    '''Read transactions from csv_file specified by filepath
    Args:
        filepath (str): the path to the file to be read
    Returns:
    list: a list of lists, where each component list is a list of string representing a transaction
    '''

    transactions = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            transactions.append(line.strip().split(',')[:-1])

    return transactions


# Initialize one-length candidate set
def createC1(transactions):
    '''Generate one-length candidates
    :param transactions: a list of lists
    :return: map of invariate constant set of one-lenght candidates
    '''

    c1 = []
    for transaction in transactions:
        for item in transaction:
            if [item] not in c1:
                c1.append([item])
    c1.sort()
    return map(frozenset, c1)


# Get k-length frequent set
def scanSet(tranSet, ck, minSupport):
    numTrans = 0
    tranSupport = {}
    cklist = list(ck)

    # for each transaction
    for tid in tranSet:
        numTrans+=1
        # for each candidate in ck
        for can in cklist:
            if can.issubset(tid):
                tranSupport[can] = tranSupport.get(can, 0) + 1
                # if not tranSupport.has_key(can):
                #    tranSupport[can]=1
                # else: tranSupport[can]+=1


    retList = []
    supportData = {}

    for key in tranSupport:
        support = tranSupport[key] / numTrans
        if support >= minSupport:
            retList.insert(0, key)
            supportData[key] = support
    # return frequent items set with the length of k, along with the corresponding support
    return retList, supportData


# Use Aprior algorithm to generate candidates for frequent itemsets
def aprioriCan(Lk, k):
    '''
    '''
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(Lk[i])[:k - 2]
            L2 = list(Lk[j])[:k - 2]
            L1.sort();
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])

    return retList


# To be implemented
def generate_frequent_itemset(transactions, minsup):
    '''Generate the frequent itemsets from transactions
    Args:
        transactions (list): a list of lists, where each component list is a list of string representing a transaction
        minsup (float): specifies the minsup for mining

    Returns:
        list: a list of frequent itemsets and each itemset is represented as a list string

    Example:
        Output: [['margarine'], ['ready soups'], ['citrus fruit','semi-finished bread'], ['tropical fruit','yogurt','coffee'], ['whole milk']]
        The meaning of the output is as follows: itemset {margarine}, {ready soups}, {citrus fruit, semi-finished bread}, {tropical fruit, yogurt, coffee}, {whole milk} are all frequent itemset

    Create candidate set
    '''

    c1 = createC1(transactions)
    tranSet = map(set, transactions)
    # print(len(list(tranSet)))

    L1, supportData = scanSet(tranSet, c1, minsup)  # get 1-length frequent set
    L = [L1]

    k = 2
    while (len(L[k - 2]) >= 0):  # until (k-1)-length frequent set is null
        ck = aprioriCan(L[k - 2], k)
        Lk, supportK = scanSet(tranSet, ck, minsup)

        # add new information of frequent set to support directory
        supportData.update(supportK)
        L.append(Lk)
        k += 1

    return L, supportData

if __name__ == "__main__":
    dataset = [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
    C1 = createC1(dataset)
    D = map(set, dataset)
    L1, supportData = scanSet(D, C1, 0.5)
    print(L1)