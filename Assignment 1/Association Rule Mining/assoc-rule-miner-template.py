import sys
import os


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


#Initialize one-length candidate set
def createC1 (transactions):
	'''Generate one-length candidates
	:param transactions: a list of lists
	:return: map of invariate constant set of one-lenght candidates
	'''
	c1 =[]
	for transaction in transactions:
		for item in transaction:
			if [item] not in c1:
				c1.append([item])

	c1.sort()
	return map(frozenset, c1)

#Get k-length frequent set
def scanSet(tranMap, ck, minSupport):
	''' Calculate support for each ck

	:param transactions: a list of lists
	:param ck: set of candidates with the length of k
	:param minSupport: minimum support
	:return:
	'''
	tranSupport = {}

	#for each transaction
	for tid in tranMap:
		#for each candidate in ck
		for can in ck:
			if can.issubset(tid):
				tranSupport[can] = tranSupport.get(can,0)+1

	numTrans = float(len(tranMap))
	retList = []
	supportData = {}
	for key in tranSupport:
		support = tranSupport[key]/numTrans
		if support >= minSupport:
			retList.insert(0, key)
		supportData[key] = support

	# return frequent items set with the length of k, along with the corresponding support
	return retList, supportData


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



	return [[]]


def main():

	if len(sys.argv) != 3 and len(sys.argv) != 4:
		print("Wrong command format, please follwoing the command format below:")
		print("python assoc-rule-miner-template.py csv_filepath minsup")
		print("python assoc-rule-miner-template.py csv_filepath minsup minconf")
		exit(0)

	
	if len(sys.argv) == 3:
		transactions = read_csv(sys.argv[1])
		result = generate_frequent_itemset(transactions, float(sys.argv[2]))

		# store frequent itemsets found by your algorithm for automatic marking
		with open('.'+os.sep+'Output'+os.sep+'frequent_itemset_result.txt', 'w') as f:
			for items in result:
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
		with open('.'+os.sep+'Output'+os.sep+'assoc-rule-result.txt', 'w') as f:
			for items in result:
				output_str = '{'
				for e in items:
					if e == '=>':
						output_str = output_str[:-1]
						output_str += '} => {'
					else:
						output_str += e
						output_str += ','

				output_str = output_str[:-1]
				output_str += '}\n'
				f.write(output_str)


main()