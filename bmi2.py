username=input()
password=input()
if username=="Kate"and password=="666666":
    print("登陆成功！")
elif username!="Kate"and password!="666666":
    username=input()
    password=input()
    if username!="Kate"and password!="666666":
        username=input()
        password=input()
        if username!="Kate"and password!="666666":
           print("3次用户名或者密码均有误！退出程序。")
        else:
            print("登陆成功！")
    else:
        print("登陆成功！")
            
