#Learn Decision Tree (ID3 Algorithm)
# Author: David R. Winer
drwiner@cs.utah.edu

#Report:
The implementation is python3.5. After setting execution privledges, run with command "./run.sh".

The features are defined as functions, and the values are enumerated in a dictionary whose key is the function to call. The features and values are provided to the ID3 algorithm at input and are used to calculate the tree. The tree is a nested dictionary whose keys are possible values for the feature represented at that node, and the values are either dictionaries or in set {True, False}. When the tree is being used (function "useTree"), we start at the root, call the function represented at that node, and return when we reach a node that is type Bool (and return that value).  

Depth is limited by providing an integer which is decremented at each recursive invocation of the ID3 algorithm during the decision tree's formation. When the depth is 1, the most-labeled output is given of the remaining samples.

Also in my implementation are tests for Majority Error and the Alien Data, which I used to test my algorithm for the previous questions, and which is not relevant to the implementation section of the homework (and therefore can be ignored).