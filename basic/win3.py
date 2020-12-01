import tkinter as tk

window = tk.Tk()
window.title('my_win3')
window.geometry('300x300')

var1 = tk.StringVar()  # 创建变量
l = tk.Label(window, bg='yellow', width=4, textvariable=var1)
l.pack()


def print_selection():
    value = lb.get(lb.curselection())
    var1.set(value)


b1 = tk.Button(window, text='print selection', width=15,
               height=2, command=print_selection)
b1.pack()

var2 = tk.StringVar()
var2.set((11, 22, 33, 44))
lb = tk.Listbox(window, listvariable=var2)
lb.pack()

# 显示主窗口
window.mainloop()
