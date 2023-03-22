# isort 테스트
import os
import time
from pathlib import Path

# from subprocess import run
# from sys import modules

# print(modules.__str__())
Path()
# print(run.__str__())
time.sleep(1)
os.times()

# black 테스트
x = {"a": 37, "b": 42, "c": 927}
very_long_variable_name = "very_long_variable_name"

if very_long_variable_name is not None and len(very_long_variable_name) > 0 or very_long_variable_name:
    z = "hello " + "world"
else:
    world = "world"
    # a = "hello {}".format(world)
    f = rf"hello {world}"


# class Foo(object):
#     def f(self):
#         return 37 * -2
#
#     def g(self, x, y=42):
#         return y


regular_formatting = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
]

# mypy 테스트
no: int = 1
print(no)


def repeat(message: str, times: int = 2) -> list:
    return [message] * times


repeat("Hi", 3)
