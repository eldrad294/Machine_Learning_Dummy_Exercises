import math
import matplotlib.pyplot as plt
x0,x1 = 2,-2
alpha = 0.01
epochs = 100
iterations = 0
cost_function = lambda x : (16 * math.pow(x,3)) - (6 * x) + 2
weight = lambda x : x - alpha * cost_function(x)
for i in range(epochs):
    temp0 = weight(x0)
    temp1 = weight(x1)
    if temp0 <= 0 or temp1 >= 0:
        break
    x0 = temp0
    x1 = temp1
    #
    plt.xlabel("P"), plt.ylabel("Cost"), plt.title("P vs Cost")
    #plt.plot(x0, cost_function(x0), 'ro'), plt.plot(x1, cost_function(x1), 'bo'), plt.show()
    iterations += 1

print("X0= " + str(x0) + " ,X1= " + str(x1) + " ,Iterations: " + str(iterations))