import numpy as np


inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, 0.8]]

weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]
# The first index of the first element in np.dot should be equal to the zeroth index of second element in np.dot like if inputs is (3,4) weights must be (4,3) because of the way 
# of applying the dot product(about matrices) so we need to transpose weights by saying "np.array(weights).T" so it turns (4,3) from (3,4).
output = np.dot(inputs, np.array(weights).T) + biases
print(output) 
print("dfawdwawd")