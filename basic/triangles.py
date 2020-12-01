def triangles():
    a=[1]
    while True:
        yield a
        a = [0]+a+[0]
        a = [a[i]+a[i+1] for i in range(len(a)-1)]
i=0
result=[]
for t in triangles():
    result.append(t)
    print(t)
    i = i + 1
    if i == 10:
        print(result)
        break