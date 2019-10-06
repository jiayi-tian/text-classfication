tempstr=input()
if tempstr[0] in['C','c']:
    f=eval(tempstr[1:-1])*1.8+32
    print("F{:.2f}".format(f))
elif tempstr[0] in['F','f']:
    c=(eval(tempstr[1:-1])-32)/1.8
    print("C{:.2f}".format(c))
else:
    print("格式错误")
