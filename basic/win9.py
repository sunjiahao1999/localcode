import tkinter as tk

window = tk.Tk()
window.title('窗口')
window.geometry('200x200')

tk.Label(window, bg='pink', width=12).pack()

frm = tk.Frame(window)
frm.pack()

# 创建小框架
frm_l = tk.Frame(frm)
frm_l.pack(side='left')
frm_r = tk.Frame(frm)
frm_r.pack(side='right')

# 放标签
tk.Label(frm_l, bg='gold', width=12).pack()
tk.Label(frm_l, bg='silver', width=12).pack()
tk.Label(frm_r, bg='brown', width=12).pack()

window.mainloop()
