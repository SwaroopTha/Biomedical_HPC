from memory_profiler import profile
import time

fp = open('./memory_profiler.log', 'w+')
@profile(stream=fp)
def my_func():
    a = [1] * (5 * 10 ** 7)
    b = [2] * (2 * 10 ** 7)
    del b
    time.sleep(1)
    del a
    time.sleep(1)
    return


# [Exercise]
# fp2 = open('./student.log', 'w+')
# @profile(stream=fp2)
# def student_func():
#     return



if __name__ == '__main__':
    my_func()

#     student_func()