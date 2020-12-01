print('hello world')


def sk(k, n):
    memo = {}

    def dp(k, n):
        nonlocal memo
        if k == 1: return n
        if n == 0: return 0
        if (k,n) in memo:
            return memo[(k,n)]

        res = float('inf')
        for i in range(1, n + 1):
            res = min(res, max(dp(k - 1, i - 1), dp(k, n - i)) + 1)
            memo[(k, n)] = res
        return res

    return dp(k, n)


a = sk(2, 100)
print(a)
