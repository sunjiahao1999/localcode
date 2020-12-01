def fib(max):
    n, a, b = 0, 0, 1
    while n < max:
        yield b
        a, b = b, a + b
        n = n + 1
    return 'done'

a = []
for n in fib(8):
    a.append(n)
print(a)

# f = fib(6)
# while True:
#     try:
#         x = next(f)
#         print(x)
#     except StopIteration as e:
#         print('Generator return value:', e.value)
#         break
