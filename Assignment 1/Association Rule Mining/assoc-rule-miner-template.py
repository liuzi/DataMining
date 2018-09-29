import os, sys
import itertools, time


# class of Hash node
class Hash_node:
    def __init__(self):
        self.children = {}  # pointer to its children
        self.Leaf_status = True  # to know the status whether current node is leaf or not
        self.bucket = {}  # contains itemsets in bucket


# class of constructing and getting hashtree
class Hash_node:
    def __init__(self):
        self.children = {}  # pointer to its children
        self.Leaf_status = True  # to know the status whether current node is leaf or not
        self.bucket = {}  # contains itemsets in bucket


# class of constructing and getting hashtree
class HashTree:
    # class constructor
    def __init__(self, max_leaf_count, max_child_count):
        self.root = Hash_node()
        self.max_leaf_count = max_leaf_count
        self.max_child_count = max_child_count
        self.frequent_itemsets = []

    # function to recursive insertion to make hashtree
    def recursively_insert(self, node, itemset, index, count):
        if index == len(itemset):
            if itemset in node.bucket:
                node.bucket[itemset] += count
            else:
                node.bucket[itemset] = count
            return

        if node.Leaf_status:  # if node is leaf
            if itemset in node.bucket:
                node.bucket[itemset] += count
            else:
                node.bucket[itemset] = count
            if len(node.bucket) == self.max_leaf_count:  # if bucket capacity increases
                for old_itemset, old_count in node.bucket.items():

                    hash_key = self.hash_function(old_itemset[index].replace(' ', ''))  # do hashing on next index

                    if hash_key not in node.children:
                        node.children[hash_key] = Hash_node()
                    self.recursively_insert(node.children[hash_key], old_itemset, index + 1, old_count)
                # since no more requirement of this bucket
                del node.bucket
                node.Leaf_status = False
        else:  # if node is not leaf
            hash_key = self.hash_function(itemset[index])
            if hash_key not in node.children:
                node.children[hash_key] = Hash_node()
            self.recursively_insert(node.children[hash_key], itemset, index + 1, count)

    def insert(self, itemset):
        itemset = tuple(itemset)
        self.recursively_insert(self.root, itemset, 0, 0)

    # to add support to candidate itemsets. Transverse the Tree and find the bucket in which this itemset is present.
    def add_support(self, itemset):
        Transverse_HNode = self.root
        itemset = tuple(itemset)
        index = 0
        while True:
            if Transverse_HNode.Leaf_status:
                if itemset in Transverse_HNode.bucket:  # found the itemset in this bucket
                    Transverse_HNode.bucket[itemset] += 1  # increment the count of this itemset.
                break
            hash_key = self.hash_function(itemset[index])
            if hash_key in Transverse_HNode.children:
                Transverse_HNode = Transverse_HNode.children[hash_key]
            else:
                break
            index += 1

    # to transverse the hashtree to get frequent itemsets with minimum support count
    #def get_frequent_itemsets(self, node, support_count, frequent_itemsets):
    def get_frequent_itemsets(self, node, minSup, frequent_itemsets, Frequent_item_value):

        if node.Leaf_status:
            for key, value in node.bucket.items():
                if value >= minSup:  # if it satisfies the condition
                    frequent_itemsets.append(list(key))  # then add it to frequent itemsets.
                    Frequent_item_value[key] = value
            return Frequent_item_value

        for child in node.children.values():
            #self.get_frequent_itemsets(child, support_count, frequent_itemsets)
            Frequent_item_value = self.get_frequent_itemsets(child, minSup, frequent_itemsets,Frequent_item_value)
        return Frequent_item_value


    # hash function for making HashTree
    def hash_function(self, val):
        return hash(val)

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


#Frequent_item_value = {}


def generate_hash_tree(candidates, max_leaf_count = 4, max_child_count = 3):
    '''
    Htree for Candidates
    :param candidates:
    :param max_leaf_count:
    :param max_child_count:
    :return:
    '''
    htree = HashTree(max_child_count, max_leaf_count)
    for itemset in candidates:
        htree.insert(itemset)
    return htree


def generate_k_subsets(transactions, length):
    '''
    :param transactions:
    :param length:
    :return:
    '''
    tran_subsets = []
    for itemset in transactions:
        tran_subsets.extend(map(list, itertools.combinations(itemset, length)))
    return tran_subsets


def subset_generation(ck, length):
    return map(list, set(itertools.combinations(ck, length)))


def generate_k_candidate(pre_frequent_set, k):
    ck = []
    # join step
    lenlk = len(pre_frequent_set)
    for i in range(lenlk):
        for j in range(i + 1, lenlk):
            L1 = list(pre_frequent_set[i])[:k - 2]
            L2 = list(pre_frequent_set[j])[:k - 2]
            if L1 == L2:
                ck.append(sorted(list(set(pre_frequent_set[i]) | set(pre_frequent_set[j]))))

    # prune step
    final_ck = []
    for candidate in ck:
        all_subsets = list(subset_generation(set(candidate), k - 1))
        found = True
        for i in range(len(all_subsets)):
            value = list(sorted(all_subsets[i]))
            if value not in pre_frequent_set:
                found = False
        if found == True:
            final_ck.append(candidate)

    return ck, final_ck


#def generate_k_frequentset(ck, minsup, filtered_transactions):
def generate_k_frequentset(ck, minSup, filtered_transactions, Frequent_item_value):

    ck_support = {}
    for tran in filtered_transactions:
        for itemset in ck:
            tran_set = set(tran)
            itemset_set = set(itemset)

            if itemset_set.issubset(tran_set):
                if tuple(itemset) not in ck_support:
                    ck_support[tuple(itemset)] = 1
                else:
                    ck_support[tuple(itemset)] += 1

    frequent_itemset = []
    for itemset in ck_support:
        if ck_support[itemset] >= minSup:
            frequent_itemset.append(sorted(list(itemset)))
            Frequent_item_value[itemset] = ck_support[itemset]

    return frequent_itemset, Frequent_item_value
    #return frequent_itemset


def generate_frequent_one_item(transactions, minSup):
    '''
    Generate frequent one itemset
    :param transactions:
    :param minSup:
    :return:
    '''
    candidate1 = {}
    Frequent_item_value = {}

    for i in range(0, len(transactions)):
        for j in range(0, len(transactions[i])):
            if transactions[i][j] not in candidate1:
                candidate1[transactions[i][j]] = 1
            else:
                candidate1[transactions[i][j]] += 1

    frequent_one_item = []
    #frequent_one_item = {}
    for item in candidate1:
        if candidate1[item] >= minSup:
            frequent_one_item = frequent_one_item + [[item]]
            Frequent_item_value[tuple([item])] = candidate1[item]
            #frequent_one_item[tuple([item])] = candidate1[item]

    #return frequent_one_item
    return frequent_one_item, Frequent_item_value


# To be implemented
def generate_frequent_itemset(transactions, minsup, max_leaf_count = 4, max_children_count = 3, withValue = False):
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
    minSup = int(minsup * float(len(transactions)))
    frequent_one_item, Frequent_item_value = generate_frequent_one_item(transactions, minSup)
    #frequent_one_item = generate_frequent_one_item(transactions, minsup)

    print(frequent_one_item)
    print(Frequent_item_value)

    # remove infrequent items from the original transactions
    filtered_transactions = []
    for i in range(0, len(transactions)):
        list_val = []
        for j in range(0, len(transactions[i])):
            if [transactions[i][j]] in frequent_one_item:
                list_val.append(transactions[i][j])
        filtered_transactions.append(list_val)

    k=2
    All_frequent_itemset = []
    All_frequent_itemset.append(0)
    All_frequent_itemset.append(frequent_one_item)


    while (len(All_frequent_itemset[k-1])>0):
        ck, final_ck = generate_k_candidate(All_frequent_itemset[k-1], k)
        print("C%d" % (k))
        print(final_ck)
        h_tree = generate_hash_tree(ck, max_leaf_count, max_children_count)
        if k>2 :
            while (len(All_frequent_itemset[k-1]) > 0):
                k_frequent_itemset, Frequent_item_value = generate_k_frequentset(final_ck, minSup, filtered_transactions, Frequent_item_value)
                #k_frequent_itemset = generate_k_frequentset(final_ck, minsup, filtered_transactions)
                All_frequent_itemset.append(k_frequent_itemset)
                print("Frequent %d item" % k)
                print(k_frequent_itemset)
                k += 1
                ck, final_ck = generate_k_candidate(All_frequent_itemset[k-1], k)
                print("C%d" % k)
                print(final_ck)
            break

        k_subsets = generate_k_subsets(filtered_transactions, k)
        for subset in k_subsets:
            h_tree.add_support(subset)
        frequen_k_item = []
        # h_tree.get_frequent_itemsets(h_tree.root, minsup, frequen_k_item)
        Frequent_item_value = h_tree.get_frequent_itemsets(h_tree.root, minSup, frequen_k_item, Frequent_item_value)
        print("Frequent %d item" % k)
        print(frequen_k_item)
        All_frequent_itemset.append(frequen_k_item)
        k += 1

    if withValue:
        return Frequent_item_value
    else:
        return All_frequent_itemset


# To be implemented
def generate_association_rules(transactions, minsup, minconf):
    '''Mine the association rules from transactions
    Args:
    	transactions (list): a list of lists, where each component list is a list of string representing a transaction
    	minsup (float): specifies the minsup for mining
    	minconf (float): specifies the minconf for mining

    Returns:
    	list: a list of association rule, each rule is represented as a list of string

    Example:
    	Output: [['root vegetables', 'rolls/buns','=>', 'other vegetables'],['root vegetables', 'yogurt','=>','other vegetables']]
    	The meaning of the output is as follows: {root vegetables, rolls/buns} => {other vegetables} and {root vegetables, yogurt} => {other vegetables} are the two associated rules found by the algorithm


    '''



    hash_map = {}
    Frequent_item_value = generate_frequent_itemset(transactions, minsup, withValue=True)
    for key, value in Frequent_item_value.items():
        hash_map[tuple(key)] = value

    All_rules = []
    for key, value in Frequent_item_value.items():
        length = len(key)
        if length == 1:
            continue

        union_support = hash_map[tuple(key)]
        for i in range(1, length):
            lefts = map(list, itertools.combinations(key, i))
            for left in lefts:
                # print("mother",key,":",hash_map[tuple(key)],"child",left,":",hash_map[tuple(left)])
                conf = 1.0 * union_support / (1.0 * hash_map[tuple(left)])
                if conf >= minconf:
                    All_rules.append([left, list(set(key) - set(left)), conf])

    return All_rules


def main():

    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print("Wrong command format, please follwoing the command format below:")
        print("python assoc-rule-miner-template.py csv_filepath minsup")
        print("python assoc-rule-miner-template.py csv_filepath minsup minconf")
        exit(0)

    if len(sys.argv) == 3:
        transactions = read_csv(sys.argv[1])
        result = generate_frequent_itemset(transactions, float(sys.argv[2]))

        with open('.' + os.sep + 'Output' + os.sep + 'frequent_itemset_result.txt', 'w') as f:
            for level in range(1,len(result)-1):
                for items in result[level]:
                    output_str = '{'
                    for e in items:
                        output_str += e
                        output_str += ','

                    output_str = output_str[:-1]
                    output_str += '}\n'
                    f.write(output_str)


    elif len(sys.argv) == 4:
        transactions = read_csv(sys.argv[1])
        minsup = float(sys.argv[2])
        minconf = float(sys.argv[3])
        result = generate_association_rules(transactions, minsup, minconf)

        # store associative rule found by your algorithm for automatic marking
        with open('.' + os.sep + 'Output' + os.sep + 'assoc-rule-result.txt', 'w') as f:
            for items in result:
                output_str = '{'
                for e in items[0]:
                    output_str += e
                    output_str += ','
                output_str = output_str[:-1] + "} => {"
                for e in items[1]:
                    output_str += e
                    output_str += ','
                output_str = output_str[:-1] + "}, "+ str(items[2]) +"\n"
                # output_str = output_str[:-1] + "}\n"

                f.write(output_str)


main()
# python assoc-rule-miner-template.py Data/Groceries.csv 0.01
# python assoc-rule-miner-template.py Data/Groceries.csv 0.01 0.2