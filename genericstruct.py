
#======================================= SYGHT, Inc. CONFIDENTIAL =================================================

import os
import numpy as np
import sys

MAXLEN = 80
PFX = '    '

class GenericStruct:
    """
    KCW version of Mike Blatchley's GenericStruct()
    All methods currently work for singly nested GenericStructs but can't have 3 layers of nested structs (yet)
    """

    def repr(self, pfx=None):
        if pfx is None: pfx = PFX
        outstr= ''
        kys= self.keys()
        kyslen = len(max(kys, key=len))
        kys.sort(key=str.lower)
        for ky in kys:
            fillstr = ' ' * (kyslen + 1 - len(ky))
            val= self.__dict__[ky]
            valstr= str(val)
            if 'GenericStruct' in str(type(val)):
                outstr += '%s%s%s\n' % (pfx, ky, fillstr)
                outstr += val.repr(PFX + pfx)[:-1]
            else:
                if isinstance(val, np.ndarray):
                    if len(val.shape) > 1:
                        valstr= "numpy array of shape %s"%str(val.shape)
                    elif len(valstr) > MAXLEN:
                        if len(val) > 4 and isinstance(val[0],(float,int,np.integer)):
                            valstr= '[%g,%g,...,%g,%g]_len=%d'%(val[0],val[1],val[-2],val[-1],len(val))
                        else:
                            valstr= '[%s,...,%s]_len=%d'%(get_rep(val[0]),get_rep(val[-1]),len(val))
                elif isinstance(val,float):
                    valstr= "%g"%(val)
                elif isinstance(val,bool):
                    valstr= "%s"%(val)
                elif isinstance(val,(int,np.integer)):
                    valstr= "%d"%(val)
                elif isinstance(val, str):
                    valstr= "\'" + val + "\'"     
                outstr += '%s%s%s%s\n' % (pfx, ky, fillstr, valstr)
        outstr += '\n'
        return outstr

    def __repr__(self):
        outstr = self.repr('    ')
        return outstr

    def get_value(self, k):
        value = None
        if k in self.__dict__:
            value = self.__dict__[k]
        return value

    def set_value(self, k, v):
        self.__dict__[k] = v

    def _get_print_str(self, attrs=None):
        """
        if attrs=None then return a short printable string including all attributes and values.   Otherwise, just include the attrs listed.
        """
        raise NotImplementedError('This does not work yet')
        param_str= ''
        if attrs == None:
            attrs= self.keys()
        for ii,attr in enumerate(attrs):
            if is_genericstruct(attr, strict=False):
                attr1,attr2= attr.split('.')
                strct= self.__dict__[attr1]
                param_str += '%s=%g\n'%(attr,strct.__dict__[attr2])
            else:
                param_str += '%s=%s\n'%(attr,repr(self.__dict__[attr]))
        return param_str

    def keys(self):
        return list(self.__dict__.keys())

    def _convert_to_dict(self):
        dct= {}
        for ky in self.keys():
            val= self.__dict__[ky]
            if 'GenericStruct' in str(type(val)):
                dct[ky]= val._convert_to_dict()
            else:
                dct[ky]= val
        return dct

    def _convert_to_list(self):
        rlst= []
        for ky in self.keys():
            val= self.__dict__[ky]
            rlst.append(val)
        return rlst

    def _import_from_dict(self, in_dct):
        for ky in in_dct.keys():
            val= in_dct[ky]
            if isinstance(val,dict):
                res= GenericStruct()
                res._import_from_dict(val)
                self.__dict__[ky]= res
            else:
                self.__dict__[ky]= in_dct[ky]

    def _new_struct(self):
        """returns a new copy of generic struct using copy.deepcopy"""
        import copy
        return copy.deepcopy(self)

    def all_vals_match_string(self, strng, verbose=False):
        """return all the values in the struct whose keys contain strng"""
        vals= []
        for ky in self.keys():
            if strng in ky:
                vals.append(self.__dict__[ky])
                if verbose:
                    print(ky, self.__dict__[ky])
        return vals

    def GetValueFromToken(self, token):
        try: value = int(token)
        except (ValueError, TypeError):
            try: value = float(token)
            except (ValueError, TypeError):
                value = token
        return value

    def read_file(self, fn, debug=False):
        if not fn: return                         # allow an empty file name
        if not os.path.isfile(fn):
            print('GenericStruct.read_file: file not found:', fn)
            return
        with open(fn, 'r') as fil:
            for line in fil:
                tokens = line.split()
                if not tokens or len(tokens) < 2 or tokens[0][0] == '#': continue # this line is either empty, invalid, or a comment
                k = tokens[0]
                if len(k) > 4 and k[-4:].lower() == '_win':
                    if sys.platform != 'win32':
                        continue                    # skip windows entry if not windows
                    k = k[0:-4]                     # else discard _win from name
                elif len(k) > 5 and k[-5:].lower() == '_unix':
                    if sys.platform == 'win32':
                        continue                    # skip unix entry if windows
                    k = k[0:-5]                     # else discard _unix from name
                if tokens[1][0] == '[':
                    value_str = line[line.index('[')+1:line.rindex(']')]
                    tokens = value_str.split()
                    v = []
                    for token in tokens:
                        value = self.GetValueFromToken(token)
                        v.append(value)
                else:
                    if tokens[1][0] == "'":
                        self.set_value(k, line.split("'")[1]) # value is a single-quoted string
                        continue
                    if tokens[1][0] == '"':
                        self.set_value(line.split('"')[1]) # value is a double-quoted string
                        continue
                    v = self.GetValueFromToken(tokens[1])
                self.set_value(k, v)
        if debug:
            print(self)


def get_rep(val):
    if isinstance(val, str):
        if len(val) > MAXLEN/4:
            return val[0:8]+'---'
    else:
        return val
    
def getParams(RLst, Param, FiltTup=None, Oper=None, ArrIdx=None):
    """Return a list of the values of Param contained in the provided list of generic structs
       If Param is array-like and Oper in ['rms','mean'] perform the operation and return list of those values
       Param can includ a "." if the targeted parameter is in a substuct
       If FiltTup is a tuple then gs.filter(RLst, FiltTup[0], FiltTup[1] is executed before extracting parameter values
    """
    param= Param
    if type(Param) is not str:
        raise ValueError('Param argument must be string')
    if FiltTup is not None:
        RLst= filter(RLst, FiltTup[0], FiltTup[1])
    params= []
    for res in RLst:
        if res == None:
            params.append(np.NaN)
            continue
        if '.' in Param:
            dlst= Param.split('.')
            if len(dlst) == 2:
                sub_struct,param= Param.split('.')
                res= res.__dict__[sub_struct]
            elif len(dlst) == 3:
                sub_struct,subsub_struct,param= Param.split('.')
                res= res.__dict__[sub_struct].__dict__[subsub_struct]
            elif len(dlst) == 4:
                sub_struct,subsub_struct,sub3_struct,param= Param.split('.')
                res= res.__dict__[sub_struct].__dict__[subsub_struct].__dict__[sub3_struct]
            else:
                raise NotImplementedError("Can't go that deep into nested structs")
        if 'genericstruct' in str(type(res)).lower() and param in res.__dict__.keys():
            if Oper != None and Oper.upper() == 'RMS':
                params.append( np.sqrt(np.mean(np.array(res.__dict__[param])*np.array(res.__dict__[param]))) )
            elif Oper != None and Oper.upper() == 'MEAN':
                params.append( np.mean(np.array(res.__dict__[param])) )
            else:
                params.append( res.__dict__[param])
        else:
            params.append(np.NaN)
    #check to see if the individual params are array-like, if so just return a specific index if appropriate
    if isinstance(params[0],(list,np.ndarray,tuple)) and ArrIdx != None:
        new_params= []
        for param in params:
            new_params.append(param[ArrIdx])
        params= new_params
    return params
        
def is_genericstruct(obj, strict=True):
    """if strict=True, will only return True if the obj has been defined as a generic struct since the last time that class was changed
       otherwise it will return True if the object was ever defined as a generic struct
    """
    if strict:
        if isinstance(obj, GenericStruct):
            return True
    else:
        if 'genericstruct' in str(type(obj)).lower():
            return True
    return False

def filter(RLst, Param, Value):
    """Return a subset of RLst, all of which have Param=Value
       Param can include one or two "." if the targeted parameter is in a substuct
    """
    nrlst= []
    param= Param
    for res in RLst:
        nres= res
        if '.' in Param:
            dlst= Param.split('.')
            if len(dlst) == 2:
                sub_struct,param= Param.split('.')
                res= res.__dict__[sub_struct]
            elif len(dlst) == 3:
                sub_struct,subsub_struct,param= Param.split('.')
                res= res.__dict__[sub_struct].__dict__[subsub_struct]
        if 'genericstruct' in str(type(res)).lower() and param in res.__dict__.keys():
            if res.__dict__[param] == Value:
                nrlst.append(nres)
    return nrlst


def verifySameness(RLst, Params):
    if type(Params) is not list:   #typically just a string if user forgets to make it a list
        Params= [Params]
    good_vals= []
    for param in Params:
        vals= getParams(RLst, param)
        if type(vals[0]) is tuple:
            raise Exception("verifySameness() doesn't work for tuples")
        uniques= np.unique(vals)
        if len(uniques) == 1:
            good_vals.append(uniques[0])
        else:
            raise Exception("All values of %s are not the same: %s"%(param,vals))
    if len(good_vals) == 1:
        return good_vals[0]
    else:
        return good_vals


def sort(RLst, Param):
    """
    sort RLst in ascending value of Param and return the new sorted list.   This makes a copy of RLst
    """
    import copy
    rlst= copy.deepcopy(RLst)
    vals= getParams(RLst, Param)
    nlst= []
    for ii in range(len(RLst)):
        jj= np.argmin(vals)
        res= rlst.pop(jj)
        nlst.append(res)
        dum= vals.pop(jj)
    return nlst


