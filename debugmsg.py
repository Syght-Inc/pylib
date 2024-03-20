from time import perf_counter

class DebugMsg:
  def __init__(self, name, enabled=False):
    self.__enabled = enabled
    self.__name = name
    self.__time = perf_counter

  def disable(self):
    self.__enabled = False

  def enable(self):
    self.__enabled = True

  def is_enabled(self):
    return self.__enabled

  def name(self, name=None):
    if name:
      self.__name = name
    return self.__name

  def msg(self, func, desc='none', *args, nl=''):
    if self.__enabled:
      d = ''.join((desc, '     '))[0:5]
      print(self.dmh(nl, d, func), args)

  def msg_always(self, func, desc='none', *args, nl=''):
    if self.__enabled:
      d = ''.join((desc, '     '))[0:5]
      print(self.dmh(nl, d, func), args)

  def msg_dbg(self, func, *args, nl=''):
    if self.__enabled:
      print(self.dmh(nl, 'debug', func), args)

  def msg_err(self, func, *args, nl=''):
    if self.__enabled:
      print(self.dmh(nl, 'ERROR', func), args)

  def msg_entry(self, func, *args, nl=''):
    if self.__enabled:
      print(self.dmh(nl, 'entry', func), args)

  def msg_exit(self, func, *args, nl=''):
    if self.__enabled:
      print(self.dmh(nl, 'exit ', func), args)

  def msg_rcvd(self, func, *args, nl=''):
    if self.__enabled:
      print(self.dmh(nl, 'rcvd ', func), args)

  def msg_sent(self, func, *args, nl=''):
    if self.__enabled:
      print(self.dmh(nl, 'sent ', func), args)

  def dmh(self, nl, desc, func):
    return '{}{: >15.6f} {} {}.{}:'.format(nl, self.__time(), desc, self.__name, func)

