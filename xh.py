n = eval(input())
for i in range(1,n+1,2):
    print("{0:^{1}}".format('*'*i, n))
