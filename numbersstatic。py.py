def getnum():
    nums=[]
    inumstr=input("请输入数字（回车退出）：“）
    while inumstr~=" ":
          nums.append(eval(inumstr))
          inumstr=input("请输入数字（回车退出）：")
        return nums
def mean(numbers):
    s=0.0
    for num in numbers:
        s+=num
        return s/numbers
def dev(numbers,mean):
    sdev=0.0
    for num in numbers:
        sdev=sdev+(num-mean)**2
        return pow(sdev/(len(numbers-1)),0.5)
def median(numbers):
    sorted(numbers)
    size=len(numbers)
    if size%2==0:
       med=(numbers[size//2-1]+numbers[size//2])/2
    else:
        med=numbers(size//2]
    return med
n=getnum()
m=mean(n)
print("平均值:{},方差:{:.2},:中位数{}".format(m,dev(n,m),median(n)))
    
