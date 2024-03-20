
#======================================= SYGHT, Inc. CONFIDENTIAL =================================================

import serial.tools.list_ports as st

class ComPorts:
  def __init__(self):
    ## List of Descriptions
    self.__descs = []
    ## List of Port Names
    self.__devices = []
    ## List of Manufacturers
    self.__mfgs = []
    ## List of Product Identifiers
    self.__pids = []
    ## List of Vendor Identifiers
    self.__vids = []

  def add(self, device, desc='NONE', mfg='NONE', pid=0, vid=0):
    self.__devices.append(device)
    self.__descs.append(desc)
    self.__mfgs.append(mfg)
    self.__pids.append(pid)
    self.__vids.append(vid)

  def scan(self):
    for port in st.comports():
      print('ComPorts.scan', port.device, port.description, port.manufacturer, port.pid, port.vid)
      self.__devices.append(port.device)
      self.__descs.append(port.description)
      self.__mfgs.append(port.manufacturer)
      self.__pids.append(port.pid)
      self.__vids.append(port.vid)

  def select(self, devs=([],[]), pids=([],[]), vids=([],[])):
    ports = []
    for i in range(len(self.__devices)):
      dev = self.__devices[i]
      pid = self.__pids[i]
      vid = self.__vids[i]
      if devs[0] and dev not in devs[0]: continue
      if pids[0] and pid not in pids[0]: continue
      if vids[0] and vid not in vids[0]: continue
      if devs[1] and dev in devs[1]: continue
      if pids[1] and pid in pids[1]: continue
      if vids[1] and vid in vids[1]: continue
      ports.append(dev)
    return ports

