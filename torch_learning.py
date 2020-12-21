import torch

# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
# # print(torch.__version__)
# x = torch.empty(5, 3)
# # print(x)
# x = torch.rand(5, 3)
# # print(x)
# x = torch.zeros(5, 3, dtype=torch.long)
# x = torch.randn(5, 3)
# x.t_()
# y = x[0, :]
# y += 1
# print(y)
# print(x[0,:])
# print(x)
# x_cp=x.clone().reshape(15)
# print(x_cp)
# x = torch.tensor([[1., 1.], [2., 3.]])
# y = x.numpy()
# print(x, y)
x = torch.ones(2, 2, requires_grad=True)
# print(x)
# print(x.grad_fn)
y = x + 2
# print(y)
z = y * y * 3
out = z.mean()
print(z.is_leaf)

