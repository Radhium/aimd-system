import numpy as np

v = np.array([1, 2, 3])
print(v)
print(v.shape)

m = np.array([[1, 2, 3],
              [4, 5, 6]])
print(m)
print(m.shape)


a = np.array([[1, 2],
              [3, 4]])

b = np.array([[5, 6],
              [7, 8]])

c = np.matmul(a, b)
print(c)
print(c.shape)



a = np.array([[1, 2, 3],
              [4, 5, 6]])

b = np.array([[1, 2, 3, 4],
              [3, 4, 3, 2],
              [4, 2, 6, 8]])

c = np.matmul(a, b)
print(c)
print(c.shape)