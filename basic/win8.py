import tkinter as tk

window = tk.Tk()
window.title('窗口')
window.geometry('200x200')

l = tk.Label(window, bg='green', text='empty', width=12)
l.pack()

# 菜单
menubar = tk.Menu(window)
filemenu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label='File', menu=filemenu)

# 加小菜单
counter = 0


def do_job():
    global counter
    l.config(text='do ' + str(counter))
    counter += 1


filemenu.add_command(label='New', command=do_job)
filemenu.add_command(label='Open', command=do_job)
filemenu.add_command(label='Save', command=do_job)
filemenu.add_separator()
filemenu.add_command(label='Exit', command=window.quit)

submenu = tk.Menu(filemenu, tearoff=0)
filemenu.add_cascade(label='Import', menu=submenu,underline=0)
submenu.add_command(label='Submenu1', command=do_job)

window.config(menu=menubar)

window.mainloop()
