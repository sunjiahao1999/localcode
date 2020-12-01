class Animal(object):
    def __init__(self):
        pass

    def run(self):
        print('Animal is running...')


class Dog(Animal):
    def run(self):
        print('dog is running')


class Cat(Animal):
    def __init__(self):
        super(Cat, self).__init__()


class Timer(object):
    def run(self):
        print('Start...')


def run_twice(bb):
    bb.run()
    bb.run()


cat = Cat()
dog = Dog()
run_twice(Timer())  # 动态语言，鸭子原则
run_twice(dog)
run_twice(Cat())
print(type(dog))
