def my_abs(x):
    if not isinstance(x, (int, float)):
        raise TypeError()
    if x >= 0:
        return x
    else:
        return -x


a = my_abs(-1)
print(a)