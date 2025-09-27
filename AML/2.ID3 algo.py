# 23BAI11010 ID3
import pandas as pd
import numpy as np

# A dictionary to represent the tree structure
def build_tree(data, features):
    target = data.columns[-1]
    
    # 1. Base case: all examples in the same class
    if len(np.unique(data[target])) == 1:
        return np.unique(data[target])[0]

    # 2. Base case: no more features to split on
    if len(features) == 0:
        return data[target].mode()[0]
    
    # Find the best feature to split on
    best_feature = find_best_feature(data, features)
    
    # Create the root node
    tree = {best_feature: {}}
    
    # Remove the chosen feature from the list of features
    remaining_features = [f for f in features if f != best_feature]
    
    # Recursively build the sub-tree for each value of the best feature
    for value in np.unique(data[best_feature]):
        sub_data = data[data[best_feature] == value].drop(columns=[best_feature])
        subtree = build_tree(sub_data, remaining_features)
        tree[best_feature][value] = subtree
    
    return tree

def find_best_feature(data, features):
    target = data.columns[-1]
    entropy_total = entropy(data[target])
    
    max_info_gain = -1
    best_feature = None
    
    for feature in features:
        info_gain = entropy_total - calculate_conditional_entropy(data, feature, target)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_feature = feature
            
    return best_feature

def entropy(column):
    # Calculate the entropy of a column
    counts = column.value_counts()
    probabilities = counts / len(column)
    return -np.sum(probabilities * np.log2(probabilities))

def calculate_conditional_entropy(data, feature, target):
    # Calculate the weighted average entropy for a given feature
    total_rows = len(data)
    weighted_entropy = 0
    
    for value in np.unique(data[feature]):
        subset = data[data[feature] == value]
        prob_subset = len(subset) / total_rows
        weighted_entropy += prob_subset * entropy(subset[target])
        
    return weighted_entropy

# Create the DataFrame
data = {'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
        'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
        'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
        'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
        'Play Tennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']}
df = pd.DataFrame(data)

# Run the ID3 algorithm
features = list(df.columns[:-1])
decision_tree = build_tree(df, features)

# Print the final tree structure
import pprint
pprint.pprint(decision_tree)

# You can now use the tree to make predictions on new data
def predict(tree, new_data):
    if not isinstance(tree, dict):
        return tree
    
    feature = list(tree.keys())[0]
    value = new_data[feature]
    
    if value in tree[feature]:
        return predict(tree[feature][value], new_data)
    else:
        # Handle cases where the value isn't in the tree (e.g., unseen data)
        return None

# Example of a new data point
new_example = {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak'}
prediction = predict(decision_tree, new_example)
print(f"\nPrediction for new data point: {prediction}")