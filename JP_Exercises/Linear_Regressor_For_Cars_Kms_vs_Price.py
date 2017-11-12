import math
#
def linear_regressor(kms):
    """ Plots a straight line fit to calculate mark based on input hours """
    # Kms vs Price
    #df = [[10, 10000], [20, 8000], [30, 7000], [40, 6000], [50, 5000], [60, 4000], [70, 3000]]
    df = [[85697,3000],[42533,7300],[84000,8000],[46863,10700],[178000,6000],[57000,6100],[97000,4000],[118000,1500],
          [199312,900],[142278,1200],[60000,7700],[31286,14000],[14822,18700],[45312,6700],[118895,4100]]
    x, y = [], []
    [(x.append(row[0]), y.append(row[1])) for row in df]
    m = (len(df) * sum([(row[0] * row[1]) for row in df]) - sum(x) * sum(y)) / (
    len(df) * sum([(math.pow(x, 2)) for x in x]) - math.pow(sum(x), 2))
    b = (sum(y) * sum([(math.pow(x, 2)) for x in x]) - sum([(row[0] * row[1]) for row in df]) * sum(x)) / (
    len(df) * sum([(math.pow(x, 2)) for x in x]) - math.pow(sum(x), 2))
    # y = mx+b
    return m*int(kms)+b
print(linear_regressor(input("Enter Amount of Kms")))