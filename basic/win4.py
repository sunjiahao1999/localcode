import tkinter as tk

window = tk.Tk()
window.title('my window')
window.geometry('200x200')

var = tk.StringVar()
l = tk.Label(window, bg='green', text='未选择', width=12)
l.pack()


def print_select():
    l.config(text='你选择了'+var.get())


r1 = tk.Radiobutton(window, text='Option A', variable=var, value='A', command=print_select)
r1.pack()
r2 = tk.Radiobutton(window, text='Option B', variable=var, value='B', command=print_select)
r2.pack()
r3 = tk.Radiobutton(window, text='Option C', variable=var, value='C', command=print_select)
r3.pack()

window.mainloop()
