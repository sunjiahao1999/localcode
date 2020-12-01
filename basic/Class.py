class Student(object):

    def __init__(self, name='lili', score='100'):
        self.__name = name
        self.__score = score

    def print_score(self):
        print('%s: %s' % (self.__name, self.__score))

    def get_score(self):
        return self.__score


bart = Student()#'Bart Simpson', 59)
lisa = Student('Lisa Simpson', 87)
bart.print_score()
lisa.print_score()
bart.age = 8
# print(lisa.__name)
print(bart.get_score())
print(bart.age)

