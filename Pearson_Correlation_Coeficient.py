import math
from scipy.stats.stats import pearsonr
import numpy as np
df = [[84,70],[76,64],[90,83],[63,45],[55,40],[60,38],[34,20]]
x, y = [], []
for item in df:
    x.append(item[0])
    y.append(item[1])
#
x_mean, y_mean = sum(x) / len(x), sum(y) / len(y)
#
upper_temp, lower_temp_x, lower_temp_y = 0,0,0
for i in range(len(df)):
    upper_temp += (x[i] - x_mean) * (y[i] - y_mean)
    lower_temp_x += math.pow(x[i] - x_mean,2)
    lower_temp_y += math.pow(y[i] - y_mean,2)
print("My value " + str(upper_temp/(math.sqrt(lower_temp_x) * math.sqrt(lower_temp_y))))
#
print("Scipy Library value " + str(pearsonr(x,y)))
#
print("Numpy Library value " + str(np.corrcoef(x,y)))
