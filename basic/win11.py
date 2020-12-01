import tkinter as tk

window = tk.Tk()
window.title('你妈死了')
window.geometry('200x200')

# tk.Label(window,text='1').pack(side='top')
# for i in range(4):
#     for j in range(4):
#         tk.Label(window,text='1').grid(row=i,column=j,ipadx=2,ipady=2)
tk.Label(window,text='1').place(x=10,y=10,anchor='nw')

window.mainloop()
