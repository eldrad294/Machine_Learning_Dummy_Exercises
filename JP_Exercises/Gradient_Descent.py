import numpy as np
import matplotlib.pyplot as plt
import math
#
def parabola(x, a, b, c, d):
    return a*x**4 - b*x**2 + c*x - d
x0,x1 = 1.5,-1.5
start_x0,start_x1 = x0,x1
alpha = 0.01
epochs = 110
iterations = 0
cost_function = lambda x : (16 * math.pow(x,3)) - (6 * x) + 2
weight = lambda x : x - alpha * cost_function(x)
#
x = np.linspace(-2, 2, 30)
y = parabola(x, 4, 3, 2, 1)
plt.xlabel("P"), plt.ylabel("Cost"), plt.title("P vs Cost")
plt.plot(x, y)
plt.plot(x0, parabola(x0, 4, 3, 2, 1), 'ro'), plt.plot(x1, parabola(x1, 4, 3, 2, 1), 'bo'), plt.plot(x, y), plt.show()
plt.show()
#
for i in range(epochs):
    temp0 = weight(x0)
    temp1 = weight(x1)
    x0 = temp0
    x1 = temp1
    #
    plt.xlabel("P"), plt.ylabel("Cost"), plt.title("P vs Cost")
    plt.plot(x0, parabola(x0, 4, 3, 2, 1), 'ro'), plt.plot(x1, parabola(x1, 4, 3, 2, 1), 'bo'), plt.plot(x, y), plt.show()
    iterations += 1
else:
    plt.xlabel("P"), plt.ylabel("Cost"), plt.title("P vs Cost")
    plt.plot(x0, parabola(x0, 4, 3, 2, 1), 'ro'), plt.plot(x1, parabola(x1, 4, 3, 2, 1), 'bo'), plt.plot(x, y), plt.show()
    print('Alpha (Rate)= '+str(alpha)+"\nepochs= "+str(epochs)+"\nx0= "+str(start_x0)+", x1= "+str(start_x1))
    print("X0= " + str(x0) + " ,X1= " + str(x1) + " ,Iterations: " + str(iterations))