import tkinter as tk
import tkinter.messagebox

window = tk.Tk()
window.title('nmsl')
window.geometry('200x200')


def hit_me():
    tk.messagebox.showinfo(title='HI', message='hahaha')
    tk.messagebox.showwarning(title='HI', message='nonono')
    tk.messagebox.showerror(title='HI', message='nmsl')
    print(tk.messagebox.askquestion(title='HI', message='hahaha')) #返回’yes‘或者’no‘
    print(tk.messagebox.askyesno(title='HI', message='hahaha')) #返回True or False
    print(tk.messagebox.askokcancel(title='HI', message='hahaha')) #返回True or False
    print(tk.messagebox.askretrycancel(title='HI', message='hahaha')) #返回True or False

tk.Button(window, text='hit me', width=12,
command = hit_me).pack()

window.mainloop()
