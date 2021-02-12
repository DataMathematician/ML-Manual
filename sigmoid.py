import matplotlib.pyplot as plt
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

z = np.arange(-10,10,0.1)

activ_func = sigmoid(z)

plt.plot(z,activ_func)
plt.axvline(0,color='red') # vertical line
plt.ylim(-0.1,1.1)
plt.xlabel('z')
plt.ylabel('$\phi (z)$') # sign as in Julia
plt.yticks([0.0,0.5,1])

# horizontal lines for .Oy
ax = plt.gca()
ax.yaxis.grid(True)
# tight
plt.tight_layout()