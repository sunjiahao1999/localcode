import tkinter as tk

window = tk.Tk()
window.title('窗口')
window.geometry('200x200')

l = tk.Label(window, bg='green', width=20, text='empty')
l.pack()
var1 = tk.IntVar()
var2 = tk.IntVar()


def print_():
    if var1.get() == 0 and var2.get() == 0:
        l.config(text='我两个都不喜欢')
    elif var1.get() == 1 and var2.get() == 0:
        l.config(text='我永远喜欢Python')
    elif var1.get() == 0 and var2.get() == 1:
        l.config(text='我永远喜欢C++')
    else:
        l.config(text='我全都要')


c1 = tk.Checkbutton(window, text='Python', variable=var1, onvalue=1,
                    offvalue=0, command=print_)
c1.pack()

c2 = tk.Checkbutton(window, text='C++', variable=var2, onvalue=1,
                    offvalue=0, command=print_)
c2.pack()

window.mainloop()
