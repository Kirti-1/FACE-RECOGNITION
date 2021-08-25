import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn')
x_data = pd.read_csv('./xdata.csv')
y_data = pd.read_csv('./ydata.csv')

x = x_data.values
y = y_data.values

x = x[:,1:]
y= y[:,1:]

y = y.reshape((-1,))
# print(x.shape)
# print(y.shape)

# Independent variables - input values
# plot the input varibles based on the output variable - dependent
# Non-linear classifier 
# plt.scatter(x[:,0], x[:,1], c = y)

# self generated query point within the range in order to check the accuracy of our classifier
query_point = np.array([2,3])

# both the query point and the dependent variables are plotted
# plt.scatter(x[:,0], x[:,1], c = y)
# plt.scatter(query_point[0], query_point[1], color = 'red')
# plt.show()

################### KNN algorithm implementation #######################

# Euclidean diatance
def dist(x1, x2):
    return np.sqrt(sum((x1-x2)**2))

def knn(x, y, query_point, k=5):
    m = x.shape[0]
    vals = []

    for i in range(m):
        d = dist(query_point, x[i])
        vals.append((d,y[i])) # (euclidean dist, label) are appended

    vals = sorted(vals)
    vals = vals[:k] # the first k shortest distance extracted along with their labels

    vals = np.array(vals)

    # unique values - duplicates removed and their count is generated
    new_vals = np.unique(vals[:,1], return_counts = True)
    print(new_vals)
    
    # index of the max number of labels from which the query_point has the shortest distance
    index = new_vals[1].argmax() 
    pred = new_vals[0][index]

    # return the predicted value 
    return pred

p = knn(x, y, query_point)
print(p)

