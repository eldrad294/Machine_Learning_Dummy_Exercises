import math
#
def linear_regressor(hours):
    """ Plots a straight line fit to calculate mark based on input hours """
    # Study Hours, Study-unit mark
    df = [[84, 70], [76, 64], [90, 83], [63, 45], [55, 40], [60, 38], [34, 20]]
    x, y = [], []
    [(x.append(row[0]), y.append(row[1])) for row in df]
    m = (len(df) * sum([(row[0] * row[1]) for row in df]) - sum(x) * sum(y)) / (
    len(df) * sum([(math.pow(x, 2)) for x in x]) - math.pow(sum(x), 2))
    b = (sum(y) * sum([(math.pow(x, 2)) for x in x]) - sum([(row[0] * row[1]) for row in df]) * sum(x)) / (
    len(df) * sum([(math.pow(x, 2)) for x in x]) - math.pow(sum(x), 2))
    # y = mx+b
    return m*int(hours)+b
print(linear_regressor(input("Enter Amount of Hours")))
#
A = 85 # Assume A is classified at 85
[(print('You need ' + str(i) + ' hours to achieve an A'), exit()) for i in range(1000) if linear_regressor(i) >= A]
#
