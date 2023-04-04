import numpy as np
X, y = np.ones((50, 1)), np.hstack(([0] * 45, [1] * 5))

print(X)
print(y)
