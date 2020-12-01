import threading
import time


def thread_job():
    print('T1 开始')
    for x in range(10):
        time.sleep(0.1)
    print('T1 结束')


def T2():
    print('T2 开始')
    print('T2 结束')


def main():
    added_thread = threading.Thread(target=thread_job, name='T1')
    T2_thread = threading.Thread(target=T2, )
    added_thread.start()
    T2_thread.start()
    # added_thread.join()
    T2_thread.join()
    print(threading.active_count())
    print(threading.enumerate())
    print(threading.current_thread())


if __name__ == '__main__':
    main()
