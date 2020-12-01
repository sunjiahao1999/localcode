# append_text='\n这是我写的文件'
# my_file=open('my_file.txt','a')
# my_file.write(append_text)
# my_file.close()

file = open('my_file.txt', 'r')
content = file.read()  # file.readline
print(content)
