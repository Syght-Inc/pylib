
#======================================= SYGHT, Inc. CONFIDENTIAL =================================================

import serial.tools.list_ports as st

class ComPorts:

  ## Format String
  FMT = '  {:15s} {:15s} {:10s} {:10s} {:25s} {}' # dev ser vid pid mfg desc

  def __init__(self):
    ## List of Descriptions
    self.__descs = []
    ## List of Port Names
    self.__devices = []
    ## List of Manufacturers
    self.__mfgs = []
    ## List of Serial Numbers
    self.__serials = []
    ## List of Product Identifiers
    self.__pids = []
    ## List of Vendor Identifiers
    self.__vids = []
    ## List of Restricted Serial Numbers
    self.__restricted_serials = []
    ## List of Restricted Serial Numbers
    self.__restricted_pids = [4026]
    ## List of Restricted Serial Numbers
    self.__restricted_vids = [1453]

  def add(self, device, desc='NONE', mfg='NONE', serial='NONE', pid=0, vid=0):
    self.__devices.append(device)
    self.__descs.append(desc)
    self.__mfgs.append(mfg)
    self.__pids.append(pid)
    self.__serials.append(serial)
    self.__vids.append(vid)

  def scan(self, quiet=True):
    if not quiet:
      print('Comports.scan:')
      print(ComPorts.FMT.format('DEVICE', 'SERIAL NUMBER', 'VENDOR ID', 'PROD ID', 'MANUFACTURER', 'DESCRIPTION'))
      port_cnt = 0
    for port in st.comports():
      if not quiet:
        print(ComPorts.FMT.format(str(port.device), str(port.serial_number), str(port.vid), str(port.pid), str(port.manufacturer), str(port.description)))
        port_cnt += 1
      self.__devices.append(port.device)
      self.__descs.append(port.description)
      self.__mfgs.append(port.manufacturer)
      self.__serials.append(port.serial_number)
      self.__pids.append(port.pid)
      self.__vids.append(port.vid)
    if not quiet and not port_cnt:
      print('  NONE')

  def select(self, devs=([],[]), sers=([],[]), pids=([],[]), vids=([],[]), quiet=True):
    """
    Select Communication Ports

    The selection critera may be device name, serial number, product identifier, or
    vendor identifier.  Each of these critera are further divided by exclusion or inclusion.
    Each selection critera parameter is a list of two lists.  The first list contains critera
    for inclusion and the second list contains critera for exclusion.  Exclusions override
    inclusions.
    @param devs   Device Name Selection Critera
    @param sers   Serial Number Selection Critera
    @param pids   Product ID Selection Critera
    @param vids   Vendor ID Selection Critera
    @param quiet  Suppress Output Flag
    @returns List of Selected Port Names
    """
    if not quiet:
      print('ComPorts.select', devs, sers, pids, vids)
    ports = []
    for i in range(len(self.__devices)):
      dev = self.__devices[i]
      ser = self.__serials[i]
      pid = self.__pids[i]
      vid = self.__vids[i]
      if not quiet:
        print('ComPorts.select dev:', dev)
      if devs[1] and dev in devs[1]: continue
      if sers[1] and ser in sers[1]: continue
      if pids[1] and pid in pids[1]: continue
      if vids[1] and vid in vids[1]: continue
      if devs[0] and dev not in devs[0]: continue
      if sers[0] and ser not in sers[0]: continue
      if pids[0] and pid not in pids[0]: continue
      if vids[0] and vid not in vids[0]: continue
      if ser in self.__restricted_serials and ser not in sers[0]: continue
      if pid in self.__restricted_pids and pid not in pids[0]: continue
      if vid in self.__restricted_vids and vid not in vids[0]: continue
      ports.append(dev)
    return ports

if __name__ == "__main__":
  cp = ComPorts()
  cp.scan(quiet=False)
