from functools import reduce


def str2float(s):
    n = s.find(".") + 1
    l = len(s)
    s = s.replace(".", "")
    f = reduce(lambda x, y: x * 10 + y, map(int, s))
    f = f / (10 ** (l - n))
    return f


print(str2float('123.456'))
    