import numpy as np
import pandas as pd

"""
The program validate the data with the decision tree saved in the dictionary
"""

data = pd.read_csv('testing__data.csv')
data = data[['Age', 'Attrition', 'BusinessTravel', 'Department', 'EducationField', 'MonthlyIncome']]

decision_tree = np.load('tree.npy').item()


def convert_value(key, value):
    if value[1:].isdigit() or value[2:].isdigit():
        if value[1:].isdigit():
            value = value[1:]
        else:
            value = value[2:]
        if key == 'Age':
            if int(value) < 38:
                value = '<38'
            else:
                value = '>=38'
        else:
            if key == 'MonthlyIncome':
                if int(value) < 5881:
                    value = '<5881'
                else:
                    value = '>=5881'
        return value
    else:
        return value


def validate(sample, current_tree):
    result = None
    for key in current_tree.keys():
        if 'Result' in current_tree.keys():
            result = current_tree['Result']
            return result
        else:
            value = sample[key]
            value = convert_value(key, value)
            result = validate(sample, current_tree[key][value])
    return result


i = 0
accurate_count = 0
tp = 0
tn = 0
fp = 0
fn = 0

while i < len(data.index):
    sample = data.iloc[i]
    if sample['Attrition'] == 'Yes' and validate(sample, decision_tree) == 'Yes':
        tp += 1
    if sample['Attrition'] == 'Yes' and validate(sample, decision_tree) == 'No':
        fn += 1
    if sample['Attrition'] == 'No' and validate(sample, decision_tree) == 'No':
        tn += 1
    if sample['Attrition'] == 'No' and validate(sample, decision_tree) == 'Yes':
        fp += 1
    if sample['Attrition'] == validate(sample, decision_tree):
        accurate_count += 1
    i += 1

print('Total:', len(data.index))
print('The Number of Right Predictions:', accurate_count)
print('Accuracy:', accurate_count / len(data.index))
print('True Positive:', tp )
print('True Negative:', tn)
print('False Negative:', fn)
print('False Positive:', fp)