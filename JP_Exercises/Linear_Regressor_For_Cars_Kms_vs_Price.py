import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#
df = [[85697, 3000], [42533, 7300], [84000, 8000], [46863, 10700], [178000, 6000], [57000, 6100], [97000, 4000],
      [118000, 1500], [199312, 900], [142278, 1200], [60000, 7700], [31286, 14000], [14822, 18700], [45312, 6700],
      [118895, 4100]]
def linear_regressor(df, kms):
    """ Plots a straight line fit to calculate mark based on input hours """
    # Kms vs Price
    x, y = [], []
    [(x.append(row[0]), y.append(row[1])) for row in df]
    m = (len(df) * sum([(row[0] * row[1]) for row in df]) - sum(x) * sum(y)) / (
    len(df) * sum([(math.pow(x, 2)) for x in x]) - math.pow(sum(x), 2))
    b = (sum(y) * sum([(math.pow(x, 2)) for x in x]) - sum([(row[0] * row[1]) for row in df]) * sum(x)) / (
    len(df) * sum([(math.pow(x, 2)) for x in x]) - math.pow(sum(x), 2))
    # y = mx+b
    return m*int(kms)+b
#
def get_r_squared(y_list):
    y_mean = sum(y_list) / len(y_list)
    denominator, numerator = 0, 0
    for y in y_list:
        denominator = denominator + math.pow(y-linear_regressor(df=df,kms=y),2)
        numerator = numerator + math.pow(y-y_mean,2)
    return 1-(numerator/denominator)
#
# Convert df into x and y points
x_list, y_list = [],[]
[x_list.append(x[0]) for x in df]
[y_list.append(y[1]) for y in df]
#
# Plot datapoints
plt.figure()
plt.xlabel('Kilometers')
plt.ylabel('Moneys')
for dp in df:
    plt.plot(dp[0], dp[1],'ro')
plt.show()
#
# Input
val = input("Enter Amount of Kms")
print(str(linear_regressor(df=df, kms=val)) + " Moneys")
print("R Squared: " + str(get_r_squared(y_list=y_list)))
#0.5932254068261312