import math
#
def linear_regressor(kms):
    """ Plots a straight line fit to calculate mark based on input hours """
    # Kms vs Price
    df = [[10, 10000], [20, 8000], [30, 7000], [40, 6000], [50, 5000], [60, 4000], [70, 3000]]
    x, y = [], []
    [(x.append(row[0]), y.append(row[1])) for row in df]
    m = (len(df) * sum([(row[0] * row[1]) for row in df]) - sum(x) * sum(y)) / (
    len(df) * sum([(math.pow(x, 2)) for x in x]) - math.pow(sum(x), 2))
    b = (sum(y) * sum([(math.pow(x, 2)) for x in x]) - sum([(row[0] * row[1]) for row in df]) * sum(x)) / (
    len(df) * sum([(math.pow(x, 2)) for x in x]) - math.pow(sum(x), 2))
    # y = mx+b
    return m*int(kms)+b
print(linear_regressor(input("Enter Amount of Kms")))