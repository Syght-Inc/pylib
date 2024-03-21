
#======================================= SYGHT, Inc. CONFIDENTIAL =================================================

from time import perf_counter

class Tracer:
  def __init__(self):
    self.__time = perf_counter
    self.__trc = []

  def clear(self):
    self.__trc = []

  def print(self, ndx=0):
    if self.__trc:
      cnt = len(self.__trc)
      if ndx < 0:
        ndx = cnt + ndx
      if ndx >= cnt - 1:
        ndx = cnt - 1
      if ndx < 0:
        ndx = 0
      print('Tracer.print ndx,cnt', ndx, cnt)
      time_prev = self.__trc[ndx][0]
      for trc in self.__trc[ndx:cnt]:
        time_this = trc[0]
        elapsed = time_this - time_prev
        time_prev = time_this
        print('{: >15.6f}'.format(elapsed), trc)

  def trace(self, *args):
    self.__trc.append((self.__time(), args))

  def trace1(self, a):
    self.__trc.append((self.__time(), a))

  def trace2(self, a, b):
    self.__trc.append((self.__time(), a, b))

  def trace3(self, a, b, c):
    self.__trc.append((self.__time(), a, b, c))

class Test:
  def __init__(self, trc):
    self.trc = trc

  def trace(self, *args):
    for arg in args:
      self.trc.trace(arg)
    self.trc.trace(args)
