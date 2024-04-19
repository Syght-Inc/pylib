
#======================================= SYGHT, Inc. CONFIDENTIAL =================================================

from time import perf_counter

class DebugMsg:
  def __init__(self, name):
    self.__enabled = False
    self.__name = name
    self.__time = perf_counter
    self.__trc = []
    self.__trc_enabled = False

  def debug_enabled(self, enabled=True):
    if enabled is not None:
      self.__enabled = enabled
    return self.__enabled

  def trace_enabled(self, enabled=True):
    if enabled is not None:
      self.__trc_enabled = enabled
    return self.__trc_enabled

  def name(self, name=None):
    if name:
      self.__name = name
    return self.__name

  def __msg(self, func, desc, *args, nl=''):
    d = ''.join((desc, '     '))[0:5]
    if self.__enabled:
      print(self.dmh(nl, d, func), args)
    if self.__trc_enabled:
      self.__trc.append((perf_counter(), d, func, args))

  def msg(self, func, desc='none', *args, nl=''):
    self.__msg(func, desc, args, nl=nl)

  def msg_always(self, func, desc='none', *args, nl=''):
    d = ''.join((desc, '     '))[0:5]
    print(self.dmh(nl, d, func), args)
    self.__trc.append((perf_counter(), d, func, args))

  def msg_dbg(self, func, *args, nl=''):
    self.__msg(func, 'debug', args, nl=nl)

  def msg_err(self, func, *args, nl=''):
    self.__msg(func, '*ERR*', args, nl=nl)

  def msg_entry(self, func, *args, nl=''):
    self.__msg(func, 'entry', args, nl=nl)

  def msg_exit(self, func, *args, nl=''):
    self.__msg(func, 'exit', args, nl=nl)

  def msg_rcvd(self, func, *args, nl=''):
    self.__msg(func, 'rcvd', args, nl=nl)

  def msg_sent(self, func, *args, nl=''):
    self.__msg(func, 'sent', args, nl=nl)

  def msg_tmo(self, func, *args, nl='', always=False):
    self.__msg(func, '*TMO*', args, nl=nl)

  def dmh(self, nl, desc, func):
    return '{}{: >15.6f} {}.{} {}:'.format(nl, self.__time(), self.__name, func, desc)

  def clear_trc(self):
    self.__trc = []

  def print_trc(self, count=None):
    """
    Print Trace Entries

    Format and print the entries in the trace list.
    """

    while True:
      first = 0
      last = len(self.__trc)
      print('DebugMsg: {} {} {}'.format(first, last, self))
      if not last:
        break
      if count is not None and count:
        if count > 0:
          if count < last:
            last = count
        else:
          first = last + count
          if first < 0:
            first = 0
      print('DebugMsg: {} {}'.format(first, last))
      time_prev = self.__trc[first][0]
      for i in range(first, last):
        trc = self.__trc[i]
        time_this = trc[0]
        time_str = '{:9.7f}'.format(time_this)
        elapsed = time_this - time_prev
        time_prev = time_this
        print('{: >15.6f}'.format(elapsed), time_str, trc[1:])
      break

  def trace(self, func, desc, *args):
    if self.__trc_enabled:
      d = ''.join((desc, '     '))[0:5]
      self.__trc.append((perf_counter(), d, func, args))
