import numpy as np
import matplotlib.pyplot as plt
def sigmoid(z):
    return 1/(1+np.exp(-z))
x=np.linspace(-5,5,100)
y=np.random.randint(0,2,100)
a=x
h=1/2*(a-y)**2
plt.plot(x,h)
plt.show()