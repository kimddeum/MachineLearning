import numpy as np

A = np.random.randint(0, 10, size=(3, 4))
B = np.random.randint(0, 10, size=(4, 3))

print("A is a 3 by 4 matrix", '\n', A, '\n', "B is a 4 by 3 matrix", '\n', B)
print("Matrix multiplication of A and B is", '\n', np.matmul(A, B))

A = np.random.randint(0, 10, size=(3, 4))
B = np.random.randint(0, 10, size=(3, 4))

print("A is a 3 by 4 matrix", '\n', A, '\n', "B is a 3 by 4 matrix", '\n', B)
print("Element-wise multiplication of A and B is", '\n', np.multiply(A, B))

A = np.random.randint(-10, 10, size=(3, 5))
b = np.random.randint(-10, 10, size=(3, 1))
x = np.random.randint(-10, 10, size=(5, 1))
y = np.zeros(3)

y = np.tanh(np.matmul(A, x) + b)

print("A = ", '\n', A)
print("b = ", '\n', b)
print("x = ", '\n', x)
print("tanh(A*x+b) = ", '\n', y)