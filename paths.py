
#======================================= SYGHT, Inc. CONFIDENTIAL =================================================

import os
import sys

class Paths:
    @property
    def paths(self):
        """
        Get System Path List
        """
        path_list = sys.path
        for path in path_list:
            print('path:', path)
        return path_list

    @paths.setter
    def paths(self, folder):
        if isinstance(folder, (list, tuple)):
            requests = folder                       # handle request for multiple folders
        else:
            requests = [folder]                     # make requested folder a list (tuple doesn't work with a single element)
        folders = []                                # list of folders already in the path
        for path in sys.path:                       # get a list of abbreviated path names
            #print('path:', path)
            if path:
                p,f = os.path.split(path)
                #print('  p:', p)
                #print('  f:', f)
                if f == 'git':
                    p,f = os.path.split(p)
                folders.append(f)
        for request in requests:
            if request not in folders:
                new_folder = os.path.join('.', request) # find the folder in the current directory
                if not os.path.isdir(new_folder):
                    new_folder = os.path.join('..', request) # find the folder in the parent directory
                    if not os.path.isdir(new_folder):
                        raise ValueError('missing: ' + request)
                sys.path.append(new_folder)

    def filename(self, folder, file, file_type, unique=False):
        new_name = os.path.join(folder, file + file_type)
        if os.path.exists(new_name):
            if not os.path.isfile(new_name):
                raise ValueError(new_name + ' is not a file')
            if unique:
                sfx = 1
                while True:
                    new_name = os.path.join(folder, file + '_%d' % sfx + file_type)
                    if not os.path.exists(new_name):
                       break                        # path does not exist so use this name
                    sfx += 1
        return new_name

    def folder(self, name, unique=False):
        if os.path.exists(name):
            if not os.path.isdir(name):
                raise ValueError(name + ' is not a folder')
            if not unique:
                return name
            sfx = 1
            while True:
                new_name = name + '_%d' % sfx
                if not os.path.exists(new_name):
                   break                        # path does not exist so use this name
                sfx += 1
        else:
            new_name = name
        os.mkdir(new_name, mode=0o777)
        return new_name
