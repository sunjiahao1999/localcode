import pickle
# a={'a':'a','b':[1,2],1:123}
# file=open('pickle_example.pickle','wb')
# pickle.dump(a,file)
# file.close()

# file =open('pickle_example.pickle','rb')
# a=pickle.load(file)
# file.close()

with open('pickle_example.pickle','rb') as file:
    a=pickle.load(file)
print(a)