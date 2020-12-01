import types


def f():
    pass


a = type(f) == types.FunctionType
print(a)
print(dir('123'))
