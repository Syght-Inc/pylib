
#======================================= SYGHT, Inc. CONFIDENTIAL =================================================

import serial
import serial.tools.list_ports as st
import sys

class ComPorts:

    ## Format String
    FMT = '  {:15s} {:10s} {:15.13s} {:10s} {:10s} {:25s} {}' # dev app ser vid pid mfg desc

    def __init__(self):
        ## List of Applications
        self.__apps = []
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
        ## List of Restricted Applications
        self.__restricted_apps = ['GIMBAL']
        ## List of Restricted Serial Numbers
        self.__restricted_serials = []
        ## List of Restricted Serial Numbers
        self.__restricted_pids = [4026]
        ## List of Restricted Serial Numbers
        self.__restricted_vids = [1453]

    def add(self, device, app='UNKNOWN', desc='NONE', mfg='NONE', serial='NONE', pid=0, vid=0, query=False, quiet=True):
        index = len(self.__apps)
        self.__apps.append(app)
        self.__devices.append(device)
        self.__descs.append(desc)
        self.__mfgs.append(mfg)
        self.__pids.append(pid)
        self.__serials.append(serial)
        self.__vids.append(vid)
        if query:
            if not quiet:
                print(ComPorts.FMT.format('DEVICE', 'APP', 'SERIAL NUMBER', 'VENDOR ID', 'PROD ID', 'MANUFACTURER', 'DESCRIPTION'))
            self.query(index, quiet)

    def allmotion(self, dev):
        is_allmotion = False
        try:
            port = serial.Serial(port=dev, timeout=0.1)
        except:
            port = None
        if port:
            port.reset_input_buffer()
            port.write(b'/1&\r\n')
            rcvd = port.readline()
            if b'EZServo' in rcvd:
                is_allmotion = True
        return is_allmotion

    def gimbal(self, dev):
        is_gimbal = False
        try:
            port = serial.Serial(port=dev, baudrate=115200, timeout=0.1)
        except:
            port = None
        if port:
            port.reset_input_buffer()
            port.write(b'v ')
            rcvd = port.readline()
            if b'Pan-Tilt Controller' in rcvd:
                is_gimbal = True
        return is_gimbal

    def syght(self, dev):
        is_syght = False
        try:
            port = serial.Serial(port=dev, baudrate=921600, parity=serial.PARITY_EVEN, stopbits=serial.STOPBITS_TWO, timeout=0.25)
        except:
            print('syght no port')
            port = None
        if port:
            port.reset_input_buffer()
            port.send_break(0.020)
            rcvd = port.readline()
            print('syght break rcvd:', rcvd)
            if b'Syght' not in rcvd:
                port.write(b'\r')
                rcvd = port.readline()
                print('syght cr rcvd:', rcvd)
            if b'Syght' in rcvd:
                is_syght = True
            port.readline()
        return is_syght
 
    def query(self, index, quiet=True):
        port = self.__devices[index]
        if self.gimbal(port):
            self.__apps[index] = 'GIMBAL'
        elif self.allmotion(port):
            self.__apps[index] = 'ALLMOTION'
        elif self.syght(port):
            self.__apps[index] = 'SYGHT'
        if not quiet:
            print(ComPorts.FMT.format(str(port), self.__apps[index], str(self.__serials[index]),
                       str(self.__vids[index]), str(self.__pids[index]), str(self.__mfgs[index]), str(self.__descs[index])))

    def scan(self, query=False, quiet=True):
        port_cnt = len(self.__devices)
        if not quiet:
            print('Comports.scan:')
            print(ComPorts.FMT.format('DEVICE', 'APP', 'SERIAL NUMBER', 'VENDOR ID', 'PROD ID', 'MANUFACTURER', 'DESCRIPTION'))
        for port in st.comports():
            self.__devices.append(port.device)
            self.__descs.append(port.description)
            self.__mfgs.append(port.manufacturer)
            self.__serials.append(port.serial_number)
            self.__pids.append(port.pid)
            self.__vids.append(port.vid)
            self.__apps.append('UNKNOWN')
            if query: 
                self.query(port_cnt, quiet)
            port_cnt += 1
        if not quiet and not port_cnt:
            print('    NONE')

    def select(self, apps=([],[]), devs=([],[]), sers=([],[]), pids=([],[]), vids=([],[]), quiet=True):
        """
        Select Communication Ports

        The selection critera may be device name, serial number, product identifier, or
        vendor identifier.    Each of these critera are further divided by exclusion or inclusion.
        Each selection critera parameter is a list of two lists.    The first list contains critera
        for inclusion and the second list contains critera for exclusion.    Exclusions override
        inclusions.
        @param devs     Device Name Selection Critera
        @param sers     Serial Number Selection Critera
        @param pids     Product ID Selection Critera
        @param vids     Vendor ID Selection Critera
        @param quiet    Suppress Output Flag
        @returns List of Selected Port Names
        """
        if not quiet:
            print('ComPorts.select', apps, devs, sers, pids, vids)
        ports = []
        for i in range(len(self.__devices)):
            app = self.__apps[i]
            dev = self.__devices[i]
            ser = self.__serials[i]
            pid = self.__pids[i]
            vid = self.__vids[i]
            if not quiet:
                print('ComPorts.select dev:', dev, app, ser, vid, pid)
            if app in apps[1]: continue
            if dev in devs[1]: continue
            if ser in sers[1]: continue
            if pid in pids[1]: continue
            if vid in vids[1]: continue
            if apps[0] and app not in apps[0]: continue
            if devs[0] and dev not in devs[0]: continue
            if sers[0] and ser not in sers[0]: continue
            if pids[0] and pid not in pids[0]: continue
            if vids[0] and vid not in vids[0]: continue
            if app in self.__restricted_apps and app not in apps[0]: continue
            if ser in self.__restricted_serials and ser not in sers[0]: continue
            if pid in self.__restricted_pids and pid not in pids[0]: continue
            if vid in self.__restricted_vids and vid not in vids[0]: continue
            ports.append(dev)
        return ports

if __name__ == "__main__":
    cp = ComPorts()
    if sys.platform != 'win32':
        cp.add('/dev/ttyTHS0', query=True, quiet=False)
    cp.scan(query=True, quiet=False)
