money=input()
if money[0]=='R':
    u=eval(money[3:])/6.78
    print("USD{:.2f}".format(u))
elif money[0]=='U':
    r=eval(money[3:])*6.78
    print("RMB{:.2f}".format(r))
