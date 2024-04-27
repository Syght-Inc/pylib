
#======================================= SYGHT, Inc. CONFIDENTIAL =================================================

import serial
import serial.tools.list_ports as st
import sys
import time

class ComPorts:

    ALLMOTION = 'ALLMOTION'
    GIMBAL = 'GIMBAL'
    SYGHT = 'SYGHT'

    GIMBAL_PID = 4026
    GIMBAL_VID = 1453

    ## Format String
    FMT = '  {:15s} {:20s} {:15.13s} {:10s} {:10s} {:25s} {}' # dev app ser vid pid mfg desc

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
        self.__restricted_apps = [ComPorts.GIMBAL]
        ## List of Restricted Product Identifiers
        self.__restricted_pids = [ComPorts.GIMBAL_PID]
        ## List of Restricted Vendor Identifiers
        self.__restricted_vids = [ComPorts.GIMBAL_VID]

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

    def syght(self, dev_index):
        dev = self.__devices[dev_index]
        try:
            port = serial.Serial(port=dev, baudrate=921600, parity=serial.PARITY_EVEN, stopbits=serial.STOPBITS_TWO, timeout=0.1)
        except:
            port = None
        app = 'UNKNOWN'
        app_token_count = 0
        try_count = 0
        while port and try_count < 2:
            try_count += 1
            port.write(b'\r')
            rcvd = port.read(1024).decode(errors='replace').upper()
            tokens = rcvd.split()
            tokens_len = len(tokens)
            try:
                index = tokens.index(ComPorts.SYGHT)
            except ValueError:
                index = -1
            if index >= 0:
                index_last = index + 1
                token_count = 1
                while index_last < tokens_len and token_count < 3:
                    if tokens[index_last] == '>':
                        break
                    index_last += 1
                    token_count += 1
                if token_count >= app_token_count:
                    app_token_count = token_count
                    app = ' '.join(tokens[index:index_last])
            if app_token_count:
                break
            if '>' in rcvd:
                app_token_count = 1
                app = ComPorts.SYGHT
                break
            port.timeout = 1.5
            port.send_break(0.25)
            time.sleep(1.0)
        self.__apps[dev_index] = app
        return app_token_count != 0

    def query(self, index, quiet=True):
        port = self.__devices[index]
        while True:
            if 'stmicro' not in self.__mfgs[index].lower():
                if ComPorts.GIMBAL not in self.__apps:
                    if self.gimbal(port):
                        self.__apps[index] = ComPorts.GIMBAL
                        break
                if self.allmotion(port):
                    self.__apps[index] = ComPorts.ALLMOTION
                    break
            if self.__vids[index] not in self.__restricted_vids and self.__pids[index] not in self.__restricted_pids:
                self.syght(index)
            break
        if not quiet:
            print(ComPorts.FMT.format(str(port), self.__apps[index], str(self.__serials[index]),
                       str(self.__vids[index]), str(self.__pids[index]), str(self.__mfgs[index]), str(self.__descs[index])))

    def scan(self, query=False, quiet=True):
        if sys.platform != 'win32':
            self.add('/dev/ttyTHS0', query=query, quiet=quiet)
        elif not quiet:
            print(ComPorts.FMT.format('DEVICE', 'APP', 'SERIAL NUMBER', 'VENDOR ID', 'PROD ID', 'MANUFACTURER', 'DESCRIPTION'))
        port_cnt = len(self.__devices)
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
            print('ComPorts.select')
            print('   apps:', apps)
            print('   devs:', devs)
            print('   sers:', sers)
            print('   pids:', pids)
            print('   vids:', vids)
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
            if app.startswith(ComPorts.SYGHT):
                app_selected = False
                for app_req in apps[0]:
                    if app_req.startswith(ComPorts.SYGHT):
                        app_selected = True
                        break
                if not app_selected: continue
            elif apps[0] and app not in apps[0]: continue
            if devs[0] and dev not in devs[0]: continue
            if sers[0] and ser not in sers[0]: continue
            if pids[0] and pid not in pids[0]: continue
            if vids[0] and vid not in vids[0]: continue
            if app in self.__restricted_apps and app not in apps[0]: continue
            ports.append(dev)
        return ports

if __name__ == "__main__":
    cp = ComPorts()
    cp.scan(query=True, quiet=False)
