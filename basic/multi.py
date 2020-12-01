import multiprocessing as mp
import threading as td
import time
import functools


def metric(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kw):
        t0 = time.time()
        res = fn(*args, **kw)
        t1 = time.time()
        print('%s executed in %s ms' % (fn.__name__, t1 - t0))
        return res

    return wrapper


def job(q):
    res = 0
    for i in range(1000000):
        res += i + i ** 2 + i ** 3
    q.put(res)  # queue


@metric
def multicore():
    q = mp.Queue()
    p1 = mp.Process(target=job, args=(q,))
    p2 = mp.Process(target=job, args=(q,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    res1 = q.get()
    res2 = q.get()
    print('multicore:', res1 + res2)


@metric
def multithread():
    q = mp.Queue()  # thread可放入process同样的queue中
    t1 = td.Thread(target=job, args=(q,))
    t2 = td.Thread(target=job, args=(q,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    res1 = q.get()
    res2 = q.get()
    print('multithread:', res1 + res2)


@metric
def normal():
    res = 0
    for _ in range(2):
        for i in range(1000000):
            res += i + i ** 2 + i ** 3
    print('normal:', res)


if __name__ == '__main__':
    normal()
    multithread()
    st2 = time.time()
    multicore()

