# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# from basic import definition as df
#
# df.my_abs(-1)
#
# import torch
#
# print(torch.randn(3, 2))
#
import numpy as np


# a = np.mat([[1, 2], [3, 4]])
# print(a)
# print(a.dot(a.I))

# a = np.zeros((2, 1))
# b = np.zeros(2)
#
# c = np.array([[1, 2],
#               [3, 4],
#               [5, 6]])
# x = map(lambda x: x + 1, b)
# print(np.array(list(x)))
# print(b[:, np.newaxis].ravel())  # 变成列向量再变回数组
# print(b @ b)
# print(c @ b)
# print(c @ a)

# a = np.array([1, 2, 3])
# b = np.array([1, 2, 3])
# c = b[:, None] @ a[None, :]
# print(c)
# a=[1]
# def wha(a):
#     b=a[0]
#     b+=1
#     return b
# print(a)
# wha(a)
# print(a)
# def serialize(a, b):
#     return np.concatenate((a.ravel(), b.ravel()))
#
#
# def deserialize(seq):
#     return seq[:2], seq[2:]
#
#
# theta1, theta2 = np.array([1, 1]), np.array([2, 4])
# theta = serialize(theta1, theta2)
#
#
# def ans(theta):
#     t1, t2 = deserialize(theta)
#     t1[0] = 0
#     t2[0] = 0
#     return serialize(t1,t2)
#
# def smd(t1,t2):
#     t1[0]=0
#     t2[0]=1
#     return
# ans(theta)
# print(theta1,theta2)
# smd(theta1,theta2)
# print(theta1,theta2)
import pandas as pd
data={'a':[]}
