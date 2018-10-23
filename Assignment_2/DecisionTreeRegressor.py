import numpy as np
import os
import json
import operator


class MyDecisionTreeRegressor():
    def __init__(self, max_depth=5, min_samples_split=1):
        '''
        Initialization
        :param max_depth: type: integer
        maximum depth of the regression tree. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.
        :param min_samples_split: type: integer
        minimum number of samples required to split an internal node:

        root: type: dictionary, the root node of the regression tree.
        '''

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def isLeave(self, obj):
        """
        函数说明:判断测试输入变量是否是一棵树
        Parameters:
            obj - 测试对象
        Returns:
            是否是一棵树
        Website:
            http://www.cuijiahua.com/
        Modify:
            2017-12-14
        """
        import types
        return (not type(obj) is dict)

    def binary_split(self,datset, feature_Index, split_value):
        mat1 = datset[np.nonzero(datset[:, feature_Index] > split_value)[0],:]
        mat0 = datset[np.nonzero(datset[:, feature_Index] <= split_value)[0],:]
        return mat0, mat1

    def get_error(self,dataset):
        if len(dataset) == 1: return 0
        else:
            return np.var(dataset[:, -1]) * np.shape(dataset)[0]

    def get_mean(self, dataset):
        return np.mean(dataset[:, -1])

    def chooseBestSplit(self, remainData):

        # If current node only has one sample, return
        if len(set(remainData[:, -1].T.tolist()[0])) < self.min_samples_split:
            return None, self.get_mean(remainData)

        m, n = remainData.shape
        S = self.get_error(remainData)
        # Error of best splitting; Index of best splitting; Attribute value of best splitting
        bestS = float('inf')
        bestIndex = None
        bestValue = None
        # Scan all attributes
        for feature_Index in range (n-1):
            for splitVal in set(remainData[:,feature_Index].T.A.tolist()[0]):

                mat0, mat1 = self.binary_split(remainData, feature_Index, splitVal)
                newS = self.get_error(mat0)+self.get_error(mat1)
                if newS < bestS:
                    bestIndex = feature_Index
                    bestValue = splitVal
                    bestS = newS
        mat0, mat1 = self.binary_split(remainData, feature_Index, splitVal)
        if (S - bestS) < 0:
            return  None, self.get_mean(remainData)
        # if (np.shape(mat0)[0] < self.min_samples_split) or (np.shape(mat1)[0] < self.min_samples_split):
        #     return None, self.get_mean(remainData)
        return bestIndex, bestValue


    def grow(self, dataset, depth):

        depth += 1
        if depth > self.max_depth: return self.get_mean(dataset)

        feature_Index, value = self.chooseBestSplit(dataset)

        if feature_Index is None: return value
        node = {}
        node['splitting_variable'] = feature_Index
        node['splitting_threshold'] = value

        left_region, right_region = self.binary_split(dataset,feature_Index,value)

        node['left'] = self.grow(left_region,depth)
        node['right'] = self.grow(right_region,depth)
        return node


    def fit(self, X, y):
        '''
        Inputs:
        X: Train feature data, type: numpy array, shape: (N, num_feature)
        Y: Train label data, type: numpy array, shape: (N,)

        You should update the self.root in this function.
        '''
        depth = 0
        self.root = self.grow(np.append(np.mat(X), np.mat(y).T, axis=1),depth)

    def search_Tree(self, vector, current_root):

        if self.isLeave(current_root):
            label = current_root
            return label
        else:
            split_variable = current_root['splitting_variable']
            split_threshold = current_root['splitting_threshold']
            if vector[split_variable] <= split_threshold:
                label = self.search_Tree(vector,current_root['left'])
            else:
                label = self.search_Tree(vector,current_root['right'])
            return label

    def predict(self, X):
        '''
        :param X: Feature data, type: numpy array, shape: (N, num_feature)
        :return: y_pred: Predicted label, type: numpy array, shape: (N,)
        '''
        if self.isLeave(self.root):
            print("Please fit the model before making prediction")
            return None
        else:
            predicted_y = np.empty((len(X),), dtype=float)
            # predicted_y = []
            for i in range(len(X)):
                label = self.search_Tree(X[i,:].T.tolist(),self.root)
                # predicted_y.append(label)
                predicted_y[i] = label
            return predicted_y


    def get_model_string(self):
        model_dict = self.root
        return model_dict

    def save_model_to_json(self, file_name):
        model_dict = self.root
        with open(file_name, 'w') as fp:
            json.dump(model_dict, fp)


# For test
if __name__=='__main__':
    for i in range(3):
        x_train = np.genfromtxt("Test_data" + os.sep + "x_" + str(i) +".csv", delimiter=",")
        y_train = np.genfromtxt("Test_data" + os.sep + "y_" + str(i) +".csv", delimiter=",")

        for j in range(2):
            tree = MyDecisionTreeRegressor(max_depth=5, min_samples_split=j + 2)
            tree.fit(x_train, y_train)

            model_string = tree.get_model_string()

            with open("Test_data" + os.sep + "decision_tree_" + str(i) + "_" + str(j) + ".json", 'r') as fp:
                test_model_string = json.load(fp)

            print(operator.eq(model_string, test_model_string))

            y_pred = tree.predict(x_train)

            y_test_pred = np.genfromtxt("Test_data" + os.sep + "y_pred_decision_tree_"  + str(i) + "_" + str(j) + ".csv", delimiter=",")
            print(np.square(y_pred - y_test_pred).mean() <= 10**-10)

