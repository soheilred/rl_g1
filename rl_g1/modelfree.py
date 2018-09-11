import numpy as np

p = 3.99
c = 3.91

Y = np.array([34,47,28])
q = 10

print(np.argmax([np.mean(p*np.minimum(Y,q) - c*q) \
            for q in range(0,100)]))
