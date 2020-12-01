import tkinter as tk
import pickle
import tkinter.messagebox

window = tk.Tk()
window.title('登录窗口')
window.geometry('500x400')

# 欢迎图片
canvas = tk.Canvas(window, width=500, height=200)
image_file = tk.PhotoImage(file='welcome.gif')
image = canvas.create_image(250, 0, anchor='n', image=image_file)
canvas.pack()

# 主界面设置
tk.Label(window, text='用户名:', font=('microsoft yahei', 12)).place(x=100, y=200)
tk.Label(window, text='密码:', font=('microsoft yahei', 12)).place(x=100, y=250)

var_name = tk.StringVar()
var_name.set('example@qq.com')
entry_name = tk.Entry(window, textvariable=var_name)
entry_name.place(x=190, y=205)
var_pwd = tk.StringVar()
entry_pwd = tk.Entry(window, textvariable=var_pwd, show='*')
entry_pwd.place(x=190, y=255)


# 登入功能设置
def login():
    usr_name = var_name.get()
    usr_pwd = var_pwd.get()

    try:
        with open('usrs_info.pickle', 'rb') as usr_file:
            usr_info = pickle.load(usr_file)
    except FileNotFoundError:
        with open('usrs_info.pickle', 'wb') as usr_file:
            usr_info = {'admin': 'admin'}
            pickle.dump(usr_info, usr_file)
    if usr_name in usr_info:  # 用户名存在
        if usr_pwd == usr_info[usr_name]:  # 密码正确
            tk.messagebox.showinfo(title='Welcome', message='你好 ' + usr_name)
        else:  # 密码错误
            tk.messagebox.showerror(message='密码错误')
    else:  # 用户名不存在
        is_sign_up = tk.messagebox.askyesno('Welcome', '你还未注册，是否现在注册？')

        if is_sign_up:
            signup()


# 注册功能设置
def signup():
    window_signup = tk.Toplevel(window)
    window_signup.geometry('350x220')
    window_signup.title('注册窗口')

    # 注册界面设置
    new_name = tk.StringVar()
    new_name.set('example@qq.com')
    tk.Label(window_signup, text='用户名：').place(x=10, y=10)
    tk.Entry(window_signup, textvariable=new_name).place(x=150, y=10)

    new_pwd = tk.StringVar()
    tk.Label(window_signup, text='密码：').place(x=10, y=50)
    tk.Entry(window_signup, textvariable=new_pwd, show='*').place(x=150, y=50)

    new_pwd_2 = tk.StringVar()
    tk.Label(window_signup, text='再次输入密码：').place(x=10, y=90)
    tk.Entry(window_signup, textvariable=new_pwd_2, show='*').place(x=150, y=90)

    # 用户名与密码审查
    def confirm():
        np = new_pwd.get()
        nn = new_name.get()
        np2 = new_pwd_2.get()

        with open('usrs_info.pickle', 'rb') as usr_file:
            usr_info = pickle.load(usr_file)

        if np != np2:  # 检查密码是否一致
            tk.messagebox.showerror('Error', '两次输入的密码不一致，请重新输入！')
            signup()
        elif nn in usr_info:  # 检查是否已有该用户
            tk.messagebox.showerror('Error', '该用户名已存在')
            signup()
        else:
            usr_info[nn] = np
            with open('usrs_info.pickle', 'wb') as usr_file:
                pickle.dump(usr_info, usr_file)
            tk.messagebox.showinfo('Welcome', '您已成功注册')
            window_signup.destroy()

    # 注册界面按钮
    tk.Button(window_signup, text='确认', command=confirm).place(x=150, y=130)


# 主界面按钮
btn_login = tk.Button(window, text='登入', font=('mircosoft yahei', 12), command=login)
btn_login.place(x=200, y=320)
btn_signup = tk.Button(window, text='注册', font=('mircosoft yahei', 12), command=signup)
btn_signup.place(x=300, y=320)

window.mainloop()
