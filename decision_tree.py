import pprint
import pandas as pd
import math
import numpy as np

"""
    The program implements the  ID3 algorithm to generate the decision tree for classification 
    and the decision tree would be save in a dictionary in hierarchy structure.
    This is not a generalize program to receive any data format. 
    It is actually only designed for finishing our assignment.
"""
attributes_values = {
            'Department': ['Research & Development', 'Sales', 'Human Resources'],
            'BusinessTravel': ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'],
            'EducationField': ['Life Sciences', 'Medical', 'Marketing'],
            'MonthlyIncome': ['<5881', '>=5881'],
            'Age': ['<38', '>=38']
        }

def split(data, attribute, value):
    """
    Splitting examples and dropping the attribute of splitting
    :param data: The examples
    :param attribute: The attribute for splitting determining how to split the example
    :param value: The value of the attribute for splitting
    :return: Return the subset of the example after splitting
    """
    subset = data.loc[data[attribute] == value]

    # drop column from current list
    subset = subset.drop([attribute], axis=1)
    return subset


def is_end_with_same_label(data):
    """
    To tell if the examples only contain ony a class of label
    :param data: the example
    :return: If the example only contains one label, then return false
    """
    # every elements are in the same class
    if 'Yes' in data['Attrition'].values and 'No' not in data['Attrition'].values:
        return 'Yes'
    else:
        if 'Yes' not in data['Attrition'].values and 'No' in data['Attrition'].values:
            return 'No'


def entropy(positive, negative, total):
    """
    To calculate the entropy. Looks stupid but works for the binary label.
    :param positive: the number positive
    :param negative: the number of negative
    :param total: the total total number positive and negative
    :return: the entropy
    """
    if positive == 0 and negative == 0:
        return 0
    if positive != 0 and negative == 0:
        return -(positive / total) * math.log((positive / total), 2)
    if positive == 0 and negative != 0:
        return -(negative / total) * math.log((negative / total), 2)
    if positive != 0 and negative != 0:
        return -(positive / total) * math.log((positive / total), 2) - (negative / total) * math.log((negative / total), 2)


def compute_info_gain(root_entropy, counts, total):
    """
    Compute the information gain for the target attribute. The function is specialized only for the attribute that
    has three
    :param root_entropy: The entropy of the splitting attribute
    :param counts: The counts consists of
                    [total number of first values, positive number of first value,
                    second number of second value, positive number of second value
                     ...]
    :param total: The total number of attributes
    :return: the information gain for the the attributes
    """
    entropy_sum = 0
    print(range(0, int(len(counts)/2)))
    i = 0
    while i < len(counts):
        entropy_sum += counts[i] / total * entropy(counts[i+1], counts[i] - counts[i+1], counts[i])
        i += 2
    info_gain = root_entropy - entropy_sum
    return info_gain


def get_category_info_gain(data, attribute, values, root_entropy, total):
    """
    Counts the label and compute the information gain
    """
    counts = []
    for value in values:
        counts.append(data.loc[data[attribute] == value].count()[attribute])
        counts.append(data.loc[data[attribute] == value].loc[data['Attrition'] == 'Yes'].count()[attribute])
    return compute_info_gain(root_entropy, counts, total)


def compute_all(data):
    """
    Compute the information gain of each attribute. The function is specialized only for our data format.
    :param data:
    :return: A dictionary of the information gain
    """
    # Terminate when there is no more attribute to be selected
    if len(data.columns) is 1:
        # return yes or no when there are only one class
        if is_end_with_same_label(data) == 'Yes' or is_end_with_same_label(data) == 'No':
            return is_end_with_same_label(data)

        # return the most common class
        yes_count = data['Attrition'].tolist().count('Yes')
        no_count = data['Attrition'].tolist().count('No')
        if yes_count > no_count:
            return 'Yes'
        else:
            return 'No'

    # terminate when all element is in the same class
    if is_end_with_same_label(data) == 'Yes' or is_end_with_same_label(data) == 'No':
        return is_end_with_same_label(data)

    root_entropy = entropy(data['Attrition'].value_counts()['Yes'],
                           data['Attrition'].value_counts()['No'],
                           len(data.index)
                           )

    result = {}
    attribute_list = data.columns.tolist()
    attribute_list.remove('Attrition')
    for key in attribute_list:
        result[key] = get_category_info_gain(data, key, attributes_values[key],
                                             root_entropy,
                                             len(data.index))
    return result


def run(data, k, tree_dict):
    """
    This is recursive function that recursively split examples and save the decision tree in the dictionary
    :param data:
    :param k: The tree depth
    :param tree_dict: The dictionary holds the decision tree
    :return: Return the current level of decision tree
    """
    k = k

    result = compute_all(data)

    # terminate when all element is in the same class
    if result == 'Yes' or result == 'No' or result is None:
        tree_dict['Result'] = result
        print(result)
        print('---------------------------------------')
    else:
        # else continue to find the maximum info and split
        max_info_gain_attr = max(result, key=result.get)

        possible_attributes = attributes_values[max_info_gain_attr]

        # Split
        for value in possible_attributes:
            print('The ' + max_info_gain_attr + ' is the max at level ' + str(k))
            print('Inspecting Condition ' + str(value))
            sub_data = split(data, max_info_gain_attr, value)

            if sub_data.size == 0:
                yes_count = data['Attrition'].tolist().count('Yes')
                no_count = data['Attrition'].tolist().count('No')
                if yes_count > no_count:
                    print('Yes')
                    if max_info_gain_attr not in tree_dict.keys():
                        tree_dict = {max_info_gain_attr: {}}
                        tree_dict[max_info_gain_attr][value] = {'Result': 'Yes'}
                    else:
                        tree_dict[max_info_gain_attr][value] = {'Result': 'Yes'}
                    print('---------------------------------------')
                else:
                    print('No')
                    if max_info_gain_attr not in tree_dict.keys():
                        # tree_dict = {max_info_gain_attr: {}}
                        tree_dict.update({max_info_gain_attr: {}})
                        tree_dict[max_info_gain_attr][value] = {'Result': 'No'}
                    else:
                        tree_dict[max_info_gain_attr][value] = {'Result': 'No'}
                    print('---------------------------------------')
            else:

                if max_info_gain_attr not in tree_dict.keys():
                    branch = {}
                    tree_dict.update({max_info_gain_attr: {value: branch}})
                    run(sub_data, k + 1, branch)
                else:
                    branch = {}
                    tree_dict[max_info_gain_attr][value] = branch
                    run(sub_data, k + 1, branch)

    return tree_dict


if __name__ == '__main__':
    data = pd.read_csv('training_data.csv')
    data = data[['Age', 'Attrition', 'BusinessTravel', 'Department', 'EducationField', 'MonthlyIncome']]
    print('Start:')
    tree_dict = {}
    run(data, 0, tree_dict)
    pprint.pprint(tree_dict)
    np.save('tree', tree_dict)
