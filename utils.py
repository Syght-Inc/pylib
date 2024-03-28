
#======================================= SYGHT, Inc. CONFIDENTIAL =================================================

import time
import numpy as np
from pylab import plt
import pickle
import os
import genericstruct as gs
COLORS= ['blue','red','green','cyan','magenta','yellow','black']*7

def timestamp(date_only=False):
    import datetime
    dum= str(datetime.datetime.now())
    dum= dum.replace(' ',', ')
    dateandtime= dum.split('.')[0]
    if date_only:
        return dateandtime.split(',')[0]
    else:
        return dateandtime
    

def GetUniqueSuffix(fname, ftype):
    """Find a numeric suffix that creates a unique path name"""
    sfx = 0
    while True:
        if ftype:
            if sfx: fpath = os.path.join(fname + '_%d' % sfx, ftype)
            else:   fpath = os.path.join(fname, ftype)
            if not os.path.exists(fpath): break # path does not exist so use this suffix
            if os.path.isdir(fpath): raise ValueError(fpath + ' IS A FOLDER')
        else:
            if sfx: fpath = fname + '_%d' % sfx
            else:   fpath = fname
            if not os.path.exists(fpath): break # path does not exist so use this suffix
            if os.path.isfile(fpath): raise ValueError(fpath + ' IS A FILE')
        sfx += 1
    return sfx


def WritePklFile(fname, obj, convert_to_dict=True, protocol=3, OvwProtect=True, RetNameOnly=False, Verbose=True, sfx=None):
    """Verbose can be a boolean or an integer.   If its an integer its the # of tabs to put before the print statment
    """
    tab_cnt= 0
    if type(Verbose) == int:
        tab_cnt= Verbose
        Verbose= True
    if fname.lower() == '.pkl': raise ValueError('INVALID FILE NAME: ' + fname)
    fpath = fname
    if fname[-4:].lower() == '.pkl': fpath = fname[0:-4] # remove '.pkl'
    if not sfx and OvwProtect:
        sfx = GetUniqueSuffix(fpath, '.pkl')
    if sfx: fpath += '_%d' % sfx
    fpath += '.pkl'                             # add file type to name
    if RetNameOnly:
        return fpath

    if Verbose:
        tabstr= '\t'*tab_cnt
        print("%sWriting to Pkl file %s"%(tabstr,fpath))
        
    dbfile = open(fpath, 'w+b')
    if convert_to_dict:
        if "GenericStruct" in str(type(obj)):
            if Verbose:
                print("converting GenericStruct(s) to dict before pickling...")
            obj= obj._convert_to_dict()
        elif isinstance(obj,list) and "GenericStruct" in str(type(obj[0])):   #assumes all items in list are dotdicts if 1st is
            nlst= []
            for res in obj:
                nlst.append( res._convert_to_dict() )
            obj= nlst
    pickle.dump(obj, dbfile, protocol=protocol)                      
    dbfile.close()
    return sfx


def ReadPklFile(fname, convert_dicts=True, Verbose=True):
    if fname[-4:] != '.pkl':
        fname += '.pkl'
    if Verbose:
        print("Reading from Pkl file %s"%fname)
    dbfile = open(fname, 'r+b')
    obj= pickle.load(dbfile)   #protocol is detected automatically
    dbfile.close()
    if convert_dicts:
        if isinstance(obj,dict):
            print("converting dict(s) to GenericStruct...")
            res= gs.GenericStruct()
            res._import_from_dict(obj)
            obj= res
        elif isinstance(obj,list) and isinstance(obj[0],dict):   #assumes all items in list are dotdicts if 1st is
            nlst= []
            for dct in obj:
                res= gs.GenericStruct()
                res._import_from_dict(dct)
                nlst.append( res )
            obj= nlst
    return obj
    
def find(condition):
    """
    This is Python3 equiv find to "from pylab import find" which was in Python 2
    """
    res, = np.nonzero(np.ravel(condition))
    return res

def moving_average(vals, n=3):
    """
    swiped from https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
    uses convolution of vals with a series of ones....it does cause some edge effects
    """
    return np.convolve(vals, np.ones(n), 'same') / n


def RootsFromPolyFit(X, Y, Degree, NumOutliers=0, Slope=None):
    """
    Get the roots from a polynomial fit to data in an array of length =Degree.  
    The array will be sorted from the smallest real roots to the largest with the imaginary roots at the end listed as np.NaN
    """
    res= Polyfit(X, Y, Degree, NumOutliers=NumOutliers, Slope=Slope)
    p= np.poly1d(res['Coeffs'])

    rts_lst= []
    for rt in p.r:
        if rt.imag == 0:
            rts_lst.insert(0,rt.real)
        else:
            rts_lst.append(np.NaN)
    rts_lst.sort()
    return np.array(rts_lst)


def Desport(vals, thresh=6, width=1, runtwice=False, verbose=False):
    """
    find any deviation that is greater than thresh*sigma away from the mean and replace it with the average of the neighboring points
    width specifies the width of the deviation in number of values 
    IMPORTANT: to safely desport width=2 you must first have desported width=1 so you have to call this method twice.  for width=3, call it 3 times,...  
    returns list or ndarray depending on type(vals)
    set runtwice=True, to rerun this since very large outliers can greatly distort the sigma value
    """
    nvals= np.array(list(vals))
    absdiffs= abs(np.diff(vals))
    idxs= find( absdiffs > thresh*np.std(vals) )
    if len(idxs) > 1:  #should see two large neighboring diff values for each single value excursion
        ii_old= idxs[0]
        for ii in idxs[1:]:
            if ii - ii_old == width and ii+1 < len(vals):
                for jj in range(width):
                    nvals[ii-jj]= (vals[ii_old] + vals[ii+1])/2.0   
                    if verbose:
                        print("At index %d: Desport() is replacing %g with %g"%(ii-jj,vals[ii-jj],nvals[ii-jj]))
            ii_old= ii         
    if type(vals) == list:
        nvals= list(nvals)
    if runtwice:
        return Desport(nvals, thresh=thresh, width=width, runtwice=False, verbose=verbose)
    return nvals


def CalcRSquared(xs, ys, ReturnMSE=False):
    """
    Calculates the square of the correlation coefficient between the xs and ys.  Often the xs are experimentally measured values and
    the ys are values from a linefit to the data.   If ReturnMSE, then also return the mean squared error
    """
    if type(xs) == list:
        xs= np.array(xs)
    if type(ys) == list:
        ys= np.array(ys)

    nn= len(xs)
    sxy= sum(xs*ys)
    sxsy= sum(xs)*sum(ys)
    numer= nn*sxy - sxsy
    b= numer/(nn*sum(xs**2) - sum(xs)**2)
    bp= numer/(nn*sum(ys**2) - sum(ys)**2)
    rsq= b*bp
    mse= 1.0*sum((xs-ys)**2)/nn
    if ReturnMSE:
        return rsq, mse
    else:
        return rsq


def Polyfit(xs, ys, degree, NumOutliers=0, Slope=None):
    """
    Returns dictionary with keys 'Coeffs' and 'Rsq'
    If NumOutliers>0 then it should be the number of data points with the bigggest squared error to be removed from the fit
    Set the value of Slope only if you want to do a linear fit with constrained slope.  For this case, I'm not sure
       this is exactly a least-squares minimized fit, but it does seem to work well.
    """
    results= {}
    if Slope == None:
        coeffs= np.polyfit(xs, ys, degree)
    else:
        xs= np.array(xs)
        ys= np.array(ys)
        b= np.mean(ys-Slope*xs)
        coeffs= np.array([Slope, b])
 
     # Polynomial Coefficients
    results['Coeffs'] = coeffs.tolist()
    p = np.poly1d(coeffs)
    results['Rsq'], results['MeanSqErr']= CalcRSquared(ys, p(xs), True)
 
    if NumOutliers > 0:
        sqerr_lst=[]
        for i_cnt, z in enumerate(xs):
            sqerr_lst.append( (p(z) - ys[i_cnt])**2 )

        x_lst=[]
        y_lst=[]
        for i_cnt, z in enumerate(xs):
            if sqerr_lst[i_cnt] < np.sort(sqerr_lst)[len(sqerr_lst)-NumOutliers]:
                x_lst.append(z)
                y_lst.append(ys[i_cnt])

        return Polyfit(x_lst, y_lst, degree, NumOutliers=0, Slope=Slope)
    return results


def GetEqString(coeffs, R_sq=None):
    """
    coeffs are the output of np.polyfit or are the res['coeffs'] output of kw.Polyfit()
    """
    ss=""
    deg=len(coeffs)-1
    for cf in coeffs:
        if deg == 0:
            ss+="%g"% cf
        elif deg == 1:
            ss+="%gx + "% cf
        else:
            ss+="%gx^%d + "% (cf, deg)
        deg-=1

    if R_sq != None:
        ss += ", R^2= %6.4f"% R_sq
    return ss


def FitLineToPlot(X, Y, Degree, PlotDataPnts=True, ShowEq=True, Coefs=None, Label="", Color='blue', Fignum=1, NumOutliers=0, Slope=None):
    """
    Returns the interpolating fit polynomial.
    if BadPnts=3, then the three data points with the largest squared error are not included in the fit.
    """
    x_arr, y_arr= RemoveFromList(list(X), 0, list(Y), 'NAN')
    y_arr, x_arr= RemoveFromList(y_arr, 0, x_arr, 'NAN')
    x_arr= np.array(x_arr)
    y_arr= np.array(y_arr)

    coeffs= Coefs
    if coeffs == None:
        dct= Polyfit(x_arr, y_arr, Degree, NumOutliers=NumOutliers, Slope=Slope)
        coeffs= dct['Coeffs']
    p= np.poly1d( coeffs )
    delx= (x_arr.max()- x_arr.min())/51.0
    xfit_arr= np.arange(x_arr.min(),x_arr.max()+delx,delx)
    yfit_arr= p(xfit_arr)
    if Fignum != None:
        plt.figure(Fignum)
        if PlotDataPnts:
            plt.plot(X,Y, 'o', color=Color, label=Label)
            plt.grid(True)
        d_lbl= ''
        if ShowEq:
            d_lbl= ": %s"%GetEqString( coeffs )
        plt.plot(xfit_arr, yfit_arr, '-', label=d_lbl, color=Color)
    if Coefs == None:
        res= gs.GenericStruct()
        res._import_from_dict(dct)
        return res

def get_serial_device(port=None, baud=115200, stopbits=2, bytesize=8, parity='N', verbose=False):
    """
    Connect to an open USB port and return the serial object.   Any timeouts must be set by calling function.
    If port is not specified, then code will start with USB0 and connect to first open port.   port can be specified as 
      '/dev/ttyUSBN' or just the integer N (and the rest of the string will be added)
    Careful!  If you don't specify the port, this method could easily steal the port from another application
       This code mainly useful when using only one serial device.
    """
    import serial
    if port == None:
        port_cnts= np.arange(4)
    elif type(port) == int:
        port_cnts= [port]
    else:
        port_cnts= [int(port[-1])]
    for pcnt in port_cnts:
        port= '/dev/ttyUSB%d'%pcnt
        try:
            serial_device= serial.Serial(port=port, baudrate=baud, bytesize=bytesize, parity=parity, stopbits=stopbits)
            if verbose:  print('Connection successful on port %s'%port)
            break
        except:
            if pcnt == port_cnts[-1]:   #no more ports to try
                raise Exception("Failed to connect to serial device")
            else:
                if verbose:
                    print('Connection failed on port %s.  Trying next one...'%port)
    return(serial_device)
    

"""*************************BEGIN GENERAL PURPOSE FUNCTIONS*************************************"""
def PackBin(value=1.0, format_str= 'f'):
    import struct, binascii
    """accept a value like a float, pack it into binary and show the hex bytes"""
    dum= struct.pack(format_str, value)
    print(binascii.hexlify(dum))

def Hexdec2bin(Number, AddSpaces=True, Width=None):
    """
    Call to np.binary_repr() with optional spacing added to output string for better readability
    Number can be either decimal or hexadecimal.
    specify width for 2-s copmliment approach to negative numbers
    see doc string for np.binary_repr()
    """
    bin_rep= np.binary_repr(Number, width=Width)
    if AddSpaces:
        bin_rep_spc= ''
        for ii,char in enumerate(bin_rep):
            jj= len(bin_rep)-ii
            if jj%4 == 0:
                bin_rep_spc += ' '
            bin_rep_spc += char
        return bin_rep_spc.strip()
    else:
        return bin_rep


def Xlinterp(xl, xu, yl, yu, x, FailOnErr=True):
    """
    Simple linear interpolation routine taken from mylib.c
    """
    xdel=xu-xl
    ydel=yu-yl
    if xdel == 0.0 or x < min(xl,xu) or x > max(xl,xu):
        msg= "Extrapolation or division by zero in Xlinterp: xl=%g xu=%g yl=%g yu=%g x=%g"%(xl, xu, yl, yu, x)
        if FailOnErr:
            raise Exception(msg)
        else:
            print(msg)
            return np.NaN
    y = yl + (ydel / xdel)*(x-xl);
    return y


def Interp1D(x, xp, fp):
    """
    Same as np.interp() except that it checks for monotonically increasing xp values where necessary and
      will reverse order of arrays when possible.
    """
    good_test= True
    if type(x) == list or type(x) == numpy.ndarray:
        pass
    else:
        x= [x]
    xp= list(xp)
    fp= list(fp)
    if np.min(x) < np.min(xp) or np.max(x) > np.max(xp):
        good_test= False
        msg= "Extreme values exceed end points in Interp1D()"      
    if xp[1] < xp[0]:
        xp.reverse()
        fp.reverse()
    for ii in range(1,len(xp)):
        if xp[ii] <= xp[ii-1]:
            msg= "x-values need to be monotonic"
            good_test= False
    if not good_test:
        raise Exception(msg)
    return np.interp(x, xp, fp)


def GetTimeStamp(DateOnly=False):
    """
    """
    tm= time.localtime()
    date= "%d-%d-%d"% (tm.tm_mon, tm.tm_mday, tm.tm_year)
    ltime= "%d:%d:%d"% (tm.tm_hour, tm.tm_min, tm.tm_sec)
    if DateOnly:
        return date
    else:
        return date +" " +ltime

def TextFileWrite(Fname, Text, Append=True):
    """Write the string Text to Fname (which will have a .txt extension applied if it does not have one)"""
    if Fname[-4:] != '.txt':
        Fname +=  '.txt'

    if Append:
        ff= open(Fname,"a")#append mode
    else:
        ff= open(Fname,"w")
    ff.write(Text)
    ff.close()
        
def CSVRead( FileName, FirstRow=0, DType=np.double, Delim=',', Verbose=(0,0)):
    """
    Reads the entire contents of the specified file into the array xx of type DType.
    FileName should be a text string.  Will skip comment lines that start with #.
    Conversion starts at line FirstRow in the file that is not a comment line.
    If DType == np.str the output is a list of lists with everything treated like a string.
    If you have a MicroPhysics-type data file first convert to all strings using this function then use CreateListOfArrays()
    to create a useful list of type-converted arrays.
    """
    import csv
    errmsg = 'crabby_pants'
    fh = open(FileName,'r')
    # do one pass through the file to determine the number of columnc and rows needed
    cr = csv.reader(fh, delimiter=Delim, skipinitialspace=True)   #I also added skipinitspc=true
    LineNum = 0
    for row in cr:
        if Verbose == 2:
            print(row)
        try:
            if row[0][0] != "#":    #skip over all comment & header lines (i.e., lines that begin with "#")
                if LineNum == FirstRow:
                    NumCols = len(row)
                LineNum += 1
        except IndexError(errmsg):
            print(row)
            raise IndexError("Error reading line %d of input file: %s"% (LineNum,errmsg))

    NumRows = LineNum - FirstRow
    tabPrint("%s: NumDataRows=%d, NumDataCols=%d"% (FileName,NumRows,NumCols), Verbose)

    # allocate the output arrays
    if NumCols > 1:
        vals = np.zeros((NumRows, NumCols), DType)
    else:
        vals = np.zeros(NumRows, DType)
    if DType == str:    
        vals = vals.tolist()    #use lists if you're dealing with strings only

    fh.seek(0)
    LineNum = 0
    ArrIndex = 0
    for row in cr:
        if row[0][0] != "#":    #skip over all comment & header lines (i.e., lines that begin with "#")
            if LineNum >= FirstRow:
                for ColIndex in range(NumCols):
                    if DType == str:
                        if NumCols > 1:
                            if ColIndex >= len(row):
                                vals[ArrIndex][ColIndex] = ""    #fill in with empty strings for short rows
                            else:
                                vals[ArrIndex][ColIndex] = row[ColIndex]
                        else:
                            vals[ArrIndex] = row[ColIndex]
                    elif DType == "mixed":
                        if NumCols > 1:
                            if ColIndex >= len(row):
                                vals[ArrIndex][ColIndex] = np.NaN    #fill in with NaN vals for short rows
                            else:
                                vals[ArrIndex][ColIndex] = CheckStringAndConvert(row[ColIndex])
                        else:
                            vals[ArrIndex] = CheckStringAndConvert(row[ColIndex])
                    else:
                        try:
                            if NumCols > 1:
                                vals[ArrIndex, ColIndex] = DType( row[ColIndex])
                            else:
                                vals[ArrIndex] = DType( row[ColIndex])
                        except:
                            fh.close()
                            ss = "Conversion error at line %d item %d in file '%s'" % (LineNum,ColIndex,FileName)
                            raise TypeError(ss)
                ArrIndex += 1
            LineNum += 1
   
    fh.close()
   
    return vals


def CSVWrite( filename, a_arr, b_vec=[], path='./', delim=',', mode='w', Verbose=(0,0)):
    """My function for writing data to a file from numpy arrays.  a_arr can be 1 or 2 dimensional. b_vec must be
    one-dimensional with length equal to number of rows of a_arr.
    """
    filename= path+filename
    tabPrint("writing to %s"%filename, Verbose)
    #mode shoulde be either "w" or "a"    check for this
    fh = open(filename, mode)
   
    NumCols= 1
    if len(a_arr.shape) == 2:
        NumRows, NumCols = a_arr.shape
    elif len(a_arr.shape) == 1:
        NumRows = len(a_arr)
    else:
        print("ERROR in myCSVWrite - array of wrong shape")
   
    for ii in range(0,NumRows):
        if NumCols == 1:
            fh.write("%g" % a_arr[ii])
        else:
            for jj in range(0,NumCols):
                if jj > 0:
                    fh.write(delim)
                fh.write("%g" % a_arr[ii][jj])
        if (len(b_vec) != 0):
            fh.write(delim + "%g" % b_vec[ii])
        fh.write("\n")

    fh.close()


def CheckStringAndConvert( strng ):
    """
    Accept a string and convert to int or float or leave it alone, whatever is best
    As written the code doesn't deal with hex values (no way to decide if '1E6' is 486 or 1,000,000)
    If empty string, just echo it back
    """
    if strng == '':
        return strng

    strng=strng.strip()  #remove leading/trailing whitespace

    if '.#IND' in strng or '#N/A' in strng:   # #N/A results from Excel na() function
        return np.NaN

    #check for exp notation: if it's a number after removing 'E,-,+' then it's a legit exp notation
    if 'E' in strng.upper():
        dstrng= strng.upper().replace('E','')
        dstrng= dstrng.replace('-','')
        dstrng= dstrng.replace('+','')
        dstrng= CheckStringAndConvert(dstrng)
        if type(dstrng) == np.int or type(dstrng) == float:  
            return float( strng )

    if strng.lower() == strng.upper() and ':' not in strng and ' ' not in strng and '/' not in strng and '\\' not in strng:   #it's probably a number
        if '.' in strng:
            newval= float( strng )
        else:
            newval= np.int( strng )
    else:
        newval=strng

    return newval


def CreateListOfArrays(up_lst, Verbose=1):
    """
    This function accepts a list of strings in the old MicroPhysics form of:
           name, val1, othername, otherval1, ...
           name, val2, othername, otherval2, ...
           name, val3, othername, otherval3, ...
           ...
    And returns a list of ndarrays.  The first item in the list is an ndarray of all the valN's.  The 2nd item in the list is an
    ndarray of all the othervalN's, ...
    """
    if len(up_lst[0])%2 != 0:
        raise Exception("List must have even number of columns")
    size= len(up_lst)
    num_cols= len(up_lst[0])/2
    tabPrint("# output columns= %d, # output rows= %d"% (num_cols, size), Level=1, Verbose=Verbose)
    out_lst= []
    name_lst= []
    for i_cnt in range(num_cols):
        d_val= CheckStringAndConvert(up_lst[0][2*i_cnt+1])
        d_arr= np.zeros(size, type(d_val))
        if type(d_val) == np.str:
            d_arr= d_arr.tolist()
        out_lst.append(d_arr)
        name_lst.append(up_lst[0][2*i_cnt])

    tabPrint(name_lst.__repr__(), Level=1, Verbose=Verbose)
    for i_cnt in range(num_cols):
        d_arr= out_lst[i_cnt]
        for j_cnt in range(size):
            d_arr[j_cnt]= CheckStringAndConvert(up_lst[j_cnt][2*i_cnt+1])

    return out_lst


def MyPlot(xs, ys, x_lbl='', y_lbl='', ln='-', mk='.', ms=None, lw=None, color=None, d_lbl='', wait=False, fignum=1, clf=False):
    if fignum == None:    return
    elif fignum == 0:     plt.figure()
    else:                 plt.figure(fignum)
    if clf:
        plt.clf()
    lntype= mk+ln
    plt.plot(xs, ys, lntype, linewidth=lw, markersize=ms, label=d_lbl, color=color)
    plt.grid(True)
    plt.xlabel(x_lbl)
    plt.ylabel(y_lbl)
    plt.legend(loc='best',fontsize=8)

    
def PlotOSPSD(RealArray, TimeStep_sec, clf=False, label="", ExtraPad=0, InitSkip=5, Color=None, LW=1, Quiet=False, IdxAvg=None, Fignum=1):
    """
    Set ExtraPad > 0 (integer only) for better amplitude accuracy
    InitSkip: skip specified # of vals which are near DC when calculating location of peak amp
    If RealArray is a 2D array it is assumed to be multiple repititions of the same data and the average ospsd will be plotted.  
    The smaller dimension of the 2D array is assumed to be number of reps to be averaged unless otherwise specified via the IdxAvg argument
    """
    wait= False
    if len(np.shape(RealArray)) > 1:
        if not Quiet:
            print("Averaging multiple data sets in frequency domain")
        nn,mm= np.shape(RealArray)
        if IdxAvg == 1 or nn > mm:
            for ii in range(mm):
                dum= ospsd(RealArray[:,ii], TimeStep_sec)
                if ii == 0:
                    kk,ll= np.shape(dum)
                    ospsds= np.zeros((kk,mm), float)
                ospsds[:,ii]= dum[:,1]
        elif IdxAvg == 0 or nn < mm:
            for ii in range(nn):
                dum= ospsd(RealArray[ii,:], TimeStep_sec)
                if ii == 0:
                    kk,ll= np.shape(dum)
                    ospsds= np.zeros((kk,nn), float)
                ospsds[:,ii]= dum[:,1]
        psd_arr= np.zeros( (kk,2), float)
        psd_arr[:,0]= dum[:,0]   #xvals
        psd_arr[:,1]= np.mean(ospsds, axis=1)   #yvals
    else:
        psd_arr= ospsd(RealArray, delta_t= TimeStep_sec, extra_pad=ExtraPad)
    print("hello",np.max(1e-6*psd_arr[:,0]))
    if np.max(1e-6*psd_arr[:,0]) > 1.0:
        MyPlot(1e-6*psd_arr[:,0], np.log10(psd_arr[:,1]/0.001), x_lbl= "Frequency  [MHz]", y_lbl= "Mag.  [dBm]", mk="", lw=LW, color=Color, \
                   d_lbl=label, wait=wait, fignum=Fignum, clf=clf)
        if max(1e-6*psd_arr[:,0]) > 1000:
            plt.xlim(xmin=0, xmax=1000)
    elif np.max(1e-3*psd_arr[:,0]) > 1.0:
        MyPlot(1e-3*psd_arr[:,0], np.log10(psd_arr[:,1]/0.001), x_lbl= "Frequency  [KHz]", y_lbl= "Mag.  [dBm]", mk="", lw=LW, color=Color, \
                   d_lbl=label, wait=wait, fignum=Fignum, clf=clf)
    else:
        MyPlot(psd_arr[:,0], np.log10(psd_arr[:,1]/0.001), x_lbl= "Frequency  [Hz]", y_lbl= "Mag.  [dBm]", mk="", lw=LW, color=Color, \
                   d_lbl=label, wait=wait, fignum=Fignum, clf=clf)
    i_max= psd_arr[InitSkip:,1].argmax()+InitSkip  
    dbm_max= np.log10(psd_arr[i_max,1]/0.001)
    frq_mhz_max= 1e-6*psd_arr[i_max,0]
    if frq_mhz_max > 1.0:
        plt.title("Max. Amp (%5.2f dBm) at Frq=%5.0fMHz"%(dbm_max, frq_mhz_max))
    else:
        plt.title("Max. Amp (%5.2f dBm) at Frq=%5.0fKHz"%(dbm_max, frq_mhz_max*1000.0))


#calculate the one-side power spectral density function of a real function
def ospsd(real_arr, delta_t=None, extra_pad=0):
    """
    Calculate the one-side power spectral density function of a real function
    If delta_t is set return a 2D array with freqs in 1st col
    extra_pad increases zero padding by a factor of 2**extra_pad.  Use more zero padding for better amplitude accuracy, but note that
       it does not help frequency resolution.
    """
    if not type(extra_pad) == int:
        raise Exception("extra_pad must be integer!!")
    size = 2
    while size < len(real_arr):
        size = size*2
    size *= 2**extra_pad
    fft_vec = np.fft.rfft(real_arr, n=size)
   
    # allocate the array
    if (delta_t == None):
        delta_t= 1
        ospsd = np.zeros(len(fft_vec), np.double)
        ospsd = 2.0*delta_t*delta_t*fft_vec*fft_vec.conjugate()
    else:
        ospsd = np.zeros((len(fft_vec),2), np.double)
        for i in range(0,len(fft_vec)):
            ospsd[i,0] = 0.5*i/(delta_t*len(fft_vec))
        ospsd[:,1] = 2.0*delta_t*delta_t*fft_vec*fft_vec.conjugate()

    return ospsd


def butter_bandpass_old(lowcut, highcut, fs, order=5):
    from scipy.signal import butter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass(lowcut, highcut, fs, order=5):
    from scipy.signal import butter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Stolen from stackoverflow.com: https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
    With improvments to avoid instability at higher orders
    lowcut, highcut and fs are the cutoff frequencies and sample frequency in Hz.
    """
    from scipy.signal import sosfilt
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    #y = sosfiltfilt(sos, data) #this supposedly removes phase error but runtime problems with scipy 0.19.1
    return y


def FilterQnD(waveform, time_step_us, cutoff_mhz):
    """
    Quick & Dirty waveform filtering (may not be strictly correct)
    """
    a_fft= np.fft.rfft(waveform)
    for ii in range(len(a_fft)):
        freq_mhz= 0.5*ii/(time_step_us*len(a_fft))
        if freq_mhz > cutoff_mhz:
            a_fft[ii] = 0.0
    return np.fft.irfft(a_fft)


def RandomVarsLowPass(N, mean=0, sigma=1, time_step_us=None, cutoff_mhz=None):
    """Return N correlated random variables from the normal distribution.  
       The LP filtering reduces the sigma so the values are multiplied by a factor to provide the desired sigma value."""
    vals= np.random.normal(loc=0, scale=sigma, size=N)
    if time_step_us != None and cutoff_mhz != None:
        newvals= FilterQnD(vals, time_step_us, cutoff_mhz)
        factor= sigma/np.std(newvals)
        vals= newvals*factor
    return vals+mean



def Just_Write_It( fh, var ):
    """
    """
    if type(var) == str:
        fh.write("%s" % var)
    elif type(var) == int:
        fh.write("%d" % var)
    else:
        fh.write("%f" % var)


def CSVWriteLists( filename, a_lst, b_lst=[], delim=',', mode='w'):
    """My function for writing data from lists to a file.  a_lst can be multi-dimensional but it must be rectangular.
    b_lst must be 1 dimensional with length equal to number of rows of a_arr.
    """

    #mode shoulde be either "w" or "a"    check for this
    fh = open(filename, mode)

    NumCols= 1
    NumRows = len(a_lst)    
    if type(a_lst[0]) == list or type(a_lst[0]) == np.ndarray:
        NumCols = len(a_lst[0])
   
    for ii in range(0,NumRows):
        if NumCols == 1:
            Just_Write_It( fh, a_lst[ii] )
        else:
            for jj in range(0,NumCols):
                if jj > 0:
                    fh.write(delim)
                Just_Write_It( fh, a_lst[ii][jj] )
        if (len(b_lst) != 0):
            fh.write(delim)
            Just_Write_It( fh, b_lst[ii])
        fh.write("\n")

    fh.close()

def PlotPDF(Mu, Sigma, NumSig=5, Norm=1, Color='blue', Label='', Fignum=1):
    """Plot a Gaussian PDF
    """
    from scipy.stats import norm
    xmax= Mu+NumSig*Sigma
    xmin= Mu-NumSig*Sigma
    xs=np.linspace(xmin, xmax, 200)
    ys= norm.pdf(xs, loc=Mu, scale=Sigma)*Norm
    plt.figure(Fignum)
    plt.plot(xs, ys, '-', color=Color, label=Label)   

def CalcCDF(Vals, BinCnt=100, Norm=True):
    xmin= min(Vals)
    xmax= max(Vals)
    xmin= min(0.95*xmin, 1.05*xmin)    #need this to account for negative values
    xmax= max(0.95*xmax, 1.05*xmax)    #need this to account for negative values
    xs= np.linspace(xmin, xmax, BinCnt)
    ys= np.zeros(BinCnt, float)
    for ii,x in enumerate(xs):
        gds= find(Vals < x)
        ys[ii]= 1.0*len(gds)
        if Norm:
            ys[ii]/= len(Vals)
    return xs, ys

def PlotCDF(Vals, BinCnt, Xs=None, Ys=None, Norm=True, Label='', Color='blue', Fignum=1):
    """Xs and Ys are the x and y values of the CDF to be plotted.   If None, then calc CDF from Vals"""
    if Xs == None or Ys == None:
        xs, ys= CalcCDF(Vals, BinCnt=BinCnt, Norm=Norm)
    else:
        xs= Xs
        ys= Ys
    plt.figure(Fignum)
    plt.plot(xs, ys, '-', label=Label, color=Color)
    plt.legend(fontsize=8, loc='upper left') # using a size in points
    plt.grid(True)
    plt.ylabel('CDF')
    xmin,xmax= plt.xlim()
    plt.hlines(0.5,xmin,xmax)
    plt.xlim(xmin,xmax)

def PlotHistogram(Vals, BinCnt=20, GaussPDF=False, RetVals=False, Fignum=1):
    xs, ys= CalcCDF(Vals, BinCnt, False)
    ysp= np.diff(ys)
    xsp= 0.5*np.diff(xs) + xs[:-1]
    plt.figure(num=Fignum)
    plt.plot(xsp, ysp, 'o-')
    if GaussPDF:
        d_lbl="Mu=%g,Sig=%g"%(np.mean(Vals),np.std(Vals))
        PlotPDF(np.mean(Vals), np.std(Vals), 3, Norm=np.mean(np.diff(xsp))*np.sum(ysp), Color='black', Label=d_lbl, Fignum=Fignum)
        plt.legend(fontsize=10) # using a size in points
    if RetVals:
        return xsp, ysp


def RollingAverage( a_arr, n, n2=None, PeakHold=False ):
    """
    Do a rolling average of n items at a time on the input array.  Can cause edge effects.
    If n2!=None then do a variable rolling average where you average over n values at the beginning of the array and you average
      over n2 values at the end of the array, the number of averages varies linearly from n to n2.
    If PeakHold, then do a rolling max(abs(a_arr)) instead of a rolling average.
    """
    import numpy as np
    in_arr= a_arr
    if type(a_arr) == list:
        in_arr= np.array(a_arr)
    out_arr= np.zeros(len(in_arr), float)
    for ii in range(len(in_arr)):
        if n2 == None:
            nn= n
        else:
            nn= int(n + (n2-n)*ii/len(in_arr) +0.5)
        ii_str= ii - (nn-1)/2
        ii_end= ii + nn/2
        if (ii_str < 0): ii_str = 0
        if (ii_end > len(in_arr)-1): ii_end = len(in_arr)-1
        if PeakHold:
            out_arr[ii]= np.max(abs(in_arr[ii_str:ii_end+1]))
        else:
            out_arr[ii]= in_arr[ii_str:ii_end+1].mean()
    return out_arr


def ReverseSortArray( array_nd ):
    dum_lst= array_nd.tolist()
    dum_lst.reverse()
    return np.array( dum_lst )


def SortTwoLists( x_lst, y_lst ):
    """
    Sort x_lst from min to max and apply same order changes to y_lst.  Works on arrays or lists.
    Return two sorted lists or arrays.
    """
    alst, blst= zip(*sorted(zip(x_lst,y_lst)))
    return alst, blst


def FindNeighbors(array, value):
    """
    Return the indices of the two nearest neigbors to value for use in interpolation.  If value is outside the range of the array, one
    of the indices returned will be an illegal index for that array (either -1 or len(array)).  
    If value==array[0], will return: -1,0
    If value==array[-1], will return: len(array)-2,len(array)-1
    Input array must be monotonically increasing.  I've only checked it out for equally spaced arrays, but I don't think this is required.
    """
    from bisect import bisect_left
    i_x= bisect_left(array, value)
    return i_x-1, i_x

def CosAlphaTheta_plusPhi(params=[], xvals=[], extra_args=[]):
    """
    For use with class LSMinFit.  
    y= amp*cos(alpha*theta + phi)
    """
    amp= params[0]
    alpha= params[1]
    phi= params[2]
    c,d,e,f= extra_args
    theta= c*xvals**3 + d*xvals**2 + e*xvals + f      #in radians
    return amp*np.cos(alpha*theta + phi)


def ExpTimesSine(params=[], xvals=[], extra_args=[]):
    """
    For use with class LSMinFit.  
    y= amp*exp(-0.5*gamma*t)*sin(w1*t + phi)
    """
    amp= params[0]
    gamma= params[1]
    f0= params[2]
    [phi]= extra_args
    w0= 2.0*np.pi*f0
    w1= np.sqrt(w0**2 - 0.25*gamma**2) 
    return amp*np.exp(-0.5*gamma*xvals)*np.sin(w1*xvals + phi)


def Triangle(params=[], xvals=[], extra_args=[]):
    """
    Triangle wave generator for use with class LSMinFit.  
    params[0]= a
    params[1]= period 
    params[2]= phase in radians
    extra_args[0] is the amplitude offset
    """
    amp= params[0]
    T= params[1]  
    phase= params[2]*T/(2*np.pi)   #param[2] is phase in radians
    amp_off= 0
    if len(extra_args) == 1:
        amp_off= extra_args[0]
    dct= Polyfit([0,T/2.0], [-amp,amp], degree=1)
    p1= np.poly1d(dct['Coeffs'])
    dct= Polyfit([T/2.0,T], [amp,-amp], degree=1)
    p2= np.poly1d(dct['Coeffs'])
    xvals= np.array((xvals+phase)%T)
    vs= np.zeros( len(xvals), float )
    idxs= np.nonzero(xvals <= T/2.0)
    vs[idxs]= p1(xvals[idxs])
    idxs= np.nonzero(xvals > T/2.0)
    vs[idxs]= p2(xvals[idxs])
    return np.array(vs) + amp_off


def Sine(params=[], xvals=[], extra_args=[]):
    """
    For use with class LSMinFit.  
    params[0]= a
    params[1]= period 
    params[2]= phase in radians
    extra_args[0] is the amplitude offset (if provided)
    y= a*sin(kx + phi) + amp_off   where k=2*pi/period
    """
    a= params[0]
    k= 2.0*np.pi/params[1]
    phi= params[2]
    amp_off= 0
    if len(extra_args) == 1:
        amp_off= extra_args[0]
    return(a*np.sin(k*xvals + phi) + amp_off)

def CosApprox(params=[], xvals=[], extra_args=[]):
    """
    For use with class LSMinFit.  
    y= a*(1 - b*theta^2)   (b=1/2 for cosine series approx truncated after 2nd order)
    theta= cx^3 + dx^2 + ex + f and c,d,e&f are provided in extra_args
    """
    a= params[0]
    b= params[1]
    c,d,e,f= extra_args
    theta= c*xvals**3 + d*xvals**2 + e*xvals + f      #in radians
    return a*(1.0 - b*(theta*theta))

def LinearTimesCos(params=[], xvals=[], extra_args=[]):
    """
    For use with class LSMinFit.  
    (ax + b)*cos(theta) where theta=cx^3 + dx^2 + ex + f and c,d,e&f are provided in extra_args
    """
    a= params[0]
    b= params[1]
    c,d,e,f= extra_args
    theta= c*xvals**3 + d*xvals**2 + e*xvals + f      #in radians
    return (a*xvals + b)*np.cos(theta)

def ConstrainedQuadFit(params=[], xvals=[], extra_args=[]):
    """
    For use with class LSMinFit.  
    Quad fit with slope=0 constraint at x=x0, where x0 is supplied in extra_args[0].
    """
    a= params[0]
    c= params[1]
    x0= extra_args[0]
    return a*xvals**2 - 2.0*a*x0*xvals + c

def CAGR(params=[], xvals=[], extra_args=[]):
    """
    AD2= AD1*CAGR^yrs
    AD2= AD1*CAGR^(days/365.25)
    days= xvals[ii] - xvals[0]
    """
    ad1= params[0]
    cagr= params[1]
    x0= xvals[0]
    return ad1*cagr**((xvals-x0)/365.25)

def AtanFit(params=[], xvals=[], extra_args=[]):
    """
    For use with class LSMinFit.  
    a= params[1],     amp= params[2]
    """
    a= params[0]
    amp= params[1]
    vals= amp*2.0*np.arctan2((xvals),a)/np.pi
    return vals

def TanhFit(params=[], xvals=[], extra_args=[]):
    """
    For use with class LSMinFit.  
    a= params[0],     amp= params[1]
    """
    #x0= params[0]
    a= params[0]
    amp= params[1]
    vals= amp*np.tanh(2*(xvals)/(np.pi*a))
    return vals

def CothFit(params=[], xvals=[], extra_args=[]):
    """For use with class LSMinFit.  """
    x0= params[0]
    a= params[1]
    y0= params[2]
    vals= y0 - 1.0 - np.cosh((xvals-x0)/a)/np.sinh((xvals-x0)/a)
    return vals

def GaussCDF(params=[], xvals=[], extra_args=[]):
    """
    For use with class LSMinFit.  
    params are the mean and sigma of a normal distribution.
    FOR THIS CODE TO WORK RIGHT YOU HAVE TO TAKE OUT ALL THE ZEROS IN THE DATA THAT IS BEING FIT
    """
    from scipy.stats import norm
    thresh= params[0]
    sig= params[1]
    return norm.cdf(thresh-xvals, loc=0, scale=sig)

def GaussCDF_FixedMean(params=[], xvals=[], extra_args=[0.0]):
    """
    For use with class LSMinFit.  
    Same as GaussCDF but with constrained mean value.
    """
    from scipy.stats import norm
    thresh= extra_args[0]
    sig= params[0]
    return norm.cdf(thresh-xvals, loc=0, scale=sig)

def GaussFit_FixedSigma(params=[], xvals=[], extra_args=[0.0]):
    """
    For use with class LSMinFit.  
    Same as GaussCDF but with constrained sigma value."""
    from scipy.stats import norm
    sig= extra_args[0]
    thresh= params[0]
    return norm.cdf(thresh-xvals, loc=0, scale=sig)

class LSMinFit:
    def __init__(self, xvals, yvals, myfunc, use_absdev=False, use_logs=False, extra_args=[], max_iter=100):
        """
        xvals and yvals are the data to be fitted
        myfunc is a function that has the functional form to be fitted with argument myfunc(params, xvals)
           see self._examplefunc below for the correct form for myfunc
        if use_absdev, then base the fit on the absolute deviation rather than least squares minimization
        if use_logs, the fit will be based on np.log10(yvals) after all zeros (log(0)=-Inf) have been removed from the data set.  This is
          useful when fitting probability functions
        """
        self.use_logs= use_logs
        if use_logs:
            if type(xvals) == np.ndarray:    self.xvals= xvals.tolist()
            else:                            self.xvals= xvals + []
            if type(yvals) == np.ndarray:    self.yvals= yvals.tolist()
            else:                            self.yvals= yvals + []
            self.yvals, self.xvals= RemoveFromList(self.yvals, 0, self.xvals)
            self.xvals= np.array(self.xvals)
            self.yvals= np.array(self.yvals)
            self.yvals= np.log10(self.yvals)
        else:
            self.xvals= xvals
            if type(xvals) == list:
                self.xvals= np.array(xvals)
            self.yvals= yvals
            if type(yvals) == list:
                self.yvals= np.array(yvals)
        self.myfunc= myfunc
        self.myfuncargs= {}
        self.myfuncargs['xvals']= self.xvals
        self.use_absdev= use_absdev
        self.extra_args= extra_args
        self.max_iter= max_iter
        self.err_arr= None

    def _examplefunc(self, params, xvals):
        """
        sample: y= ax^2 + c
        """
        a= params[0]
        c= params[1]
        return a*xvals**2 + c

    def SqErr(self, params, xvals, extra_args):
        """
        """
        self.myfuncargs['params']= params
        self.myfuncargs['extra_args']= extra_args
        if self.use_logs:
            err= self.yvals - np.log10( self.myfunc(**self.myfuncargs) )
        else:
            err= self.yvals - self.myfunc(**self.myfuncargs)
        self.err_arr= err*err
        return np.sum(self.err_arr)

    def AbsDev(self, params, xvals, extra_args):
        """
        """
        self.myfuncargs['params']= params
        self.myfuncargs['extra_args']= extra_args
        if self.use_logs:
            err= self.yvals - np.log10( self.myfunc(**self.myfuncargs) )
        else:
            err= self.yvals - self.myfunc(**self.myfuncargs)
        self.err_arr= np.abs(err)
        return np.sum(self.err_arr)

    def GetLSProb_Fit(self, param_guesses, ret_err=False, Quiet=False):
        """
        if ret_err=True, return the error array the sum of which is what was minimized
        """
        from scipy.optimize import fmin
        out_flag= (Quiet==False)
        if self.use_absdev:
            fit_par= fmin(self.AbsDev,param_guesses,args=(self.xvals,self.extra_args),maxiter=self.max_iter,full_output=True,disp=out_flag) #note args is a tuple
        else:
            fit_par= fmin(self.SqErr,param_guesses,args=(self.xvals,self.extra_args),maxiter=self.max_iter,full_output=True,disp=out_flag) #note args is a tuple
        iter_performed= fit_par[2]
        if not Quiet:  print("\nMin Err=", fit_par[1])
        if iter_performed >= self.max_iter:
            raise Exception("Too many iterations performed in GetLSProb_Fit()")
        if ret_err:
            return fit_par[0], self.err_arr
        else:
            return fit_par[0]


def RemoveFromList(Lst, Val, Lst2=None, Operation="EQUAL"):
    """
    Default: remove all members of Lst that equal Val and return what's left of the list.  If Lst2 is given then remove the same indices from
    Lst2 that you removed from Lst.
    e.g.,: RemoveFromList(range(5), 3, Lst2=range(5,10)) returns ([0, 1, 2, 4], [5, 6, 7, 9])
    Operation must be:
       "GREATER" then remove all members of Lst that are > Val.
       "GREATER_EQUAL" then remove members that are >= Val.
       "LESS" than remove all members of Lst that are < Val.
       "LESS_EQUAL" then remove members that are <= Val.
       "NAN" then remove all members that equal None or np.NaN
    NOTE: For ndarrays probably can do this faster using pylab.find().
    """
    import copy
    Operation= Operation.upper()
    if "EQUAL" not in Operation and "GREATER" not in Operation and "LESS" not in Operation and "NAN" not in Operation:
        raise Exception("Incorrect Operation in RemoveFromList()")

    dlst= copy.deepcopy(Lst)
    if Lst2 != None:
        if len(Lst) != len(Lst2):
            raise Exception("Lists must be of same length in RemoveFromList()")
        dlst2= copy.deepcopy(Lst2)
    last_ii= len(dlst)-1
    ii=0
    while ii<=last_ii:
        if ("EQUAL" in Operation and dlst[ii] == Val) or ("GREATER" in Operation and dlst[ii] > Val) \
                or ("LESS" in Operation and dlst[ii] < Val) or ("NAN" in Operation and (dlst[ii] == None or NaN_Test(dlst[ii]))):
            dlst.pop(ii)
            last_ii-= 1
            if Lst2 != None:
                dlst2.pop(ii)
        else:
            ii+=1
    if Lst2 == None:    
        return dlst
    else:
        return dlst, dlst2


def SelectArrayValues(Arr1, mask, Arr2=None):
    """
    Return new Arr1 with only the elements that satisfy the condition in mask.  If Arr2 is specified then it must be the same size
      as Arr1 and the same indices will be returned (i.e., the condition is only applied to Arr1)
    Example 1:
      dum1= np.arange(10)
      dum2= np.arange(10,20)
      mask= Arr1 > 5
      new1, new2= SeltectArrayValues(dum1, dum1>5, dum2)
      result:
        new1= array([6, 7, 8, 9])
        new2= array([16, 17, 18, 19])
    Example 2:
      SelectArrayValues(vals, (vals >=0) | (vals < 0))   #removes np.NaN values from ndarray vals
    """
    new1= np.extract(mask, Arr1)
    if Arr2 == None:
        return new1
    else:
        new2= np.extract(mask, Arr2)
        return new1, new2


def RemoveFromArrays(Arr, Val, Arr2=None, Verbose=(0,0)):
    """
    Remove all members of Arr that equal Val and return the new smaller array.  If Arr2 is given it must be of the same len at Arr and
      it will have the same indices removed as were removed from Arr.  This is much faster than RemoveFromList for large arrays.
    """
    import copy
    print("\n****DEPRECATED CODE.  Use SelectArrayValues instead****\n")
    bad_is= find( Arr==Val )
    new_arr= np.zeros(len(Arr)-len(bad_is), type(Arr[0]))
    if Arr2 != None:
        new_arr2= np.zeros(len(new_arr), type(Arr2[0]))
    bidx= 0
    jj= 0
    for ii in range(len(Arr)):
        if bidx < len(bad_is) and ii == bad_is[bidx]:
            bidx += 1
        else:
            new_arr[jj]= Arr[ii]
            if Arr2 != None:
                new_arr2[jj]= Arr2[ii]
            jj+=1
    tabPrint("%d bad values removed"%len(bad_is), Verbose)
    if Arr2 != None:
        return new_arr, new_arr2
    else:
        return new_arr


def ExtractFromArray(Arr, Val, Arr2=None, Operation='EQUAL'):
    """
    Return an new array containing only the values for "Arr Operation Val".  Operation can be 'EQUAL', 'GREATER', 'LESS', ...
    If Arr2 is provided return the values from the same indices that were removed from Arr.  
    Operation is only on Arr.
    Arr and Arr2 must be the same size."""
    if Operation == 'EQUAL':
        condition= Arr == Val
    else:
        raise Exception('Code not written yet')
    ret_arr= np.extract(condition, Arr)
    if Arr2 != None:
        ret_arr2= np.extract(condition, Arr2)
        return ret_arr, ret_arr2
    else:
        return ret_arr
   

def Difference(xs, ys):
    derivs= np.zeros(len(ys), float)
    for ii in range(1,len(ys)-1,1):
        numer= ys[ii+1] - ys[ii-1]
        denom= xs[ii+1] - xs[ii-1]
        derivs[ii]= numer/denom
    return derivs


def ListMultiValuePop(Lst, BadIndices):
    bads= list(BadIndices)
    bads.sort(reverse=True)
    for ii in bads:
        Lst.pop(ii)


def NaN_Test(value):
    """TODO:  Switch to using math.isnan() function
    """
    if ((value > 0) == False) and ((value < 0) == False) and ((value == 0) == False):
        return True
    else:
        return False

def CalcProfileWidth(Xvals, Yvals, Where=0.5):
    """
    Calculate the width of a profile (U-shaped or inverted-U-shaped).  If Where=0.5 width is calculate at half
    peak value (i.e., x=ymin + 0.5*(ymax-ymin)).  The profile function must be well behaved.
    If Xvals or Yvals == None, then use the class attributes
    Return ctr and wid
    """
    minval= min(Yvals)
    maxval= max(Yvals)
    thresh=minval + Where*(maxval-minval)

    i_cnt= 0
    while (Yvals[i_cnt]-thresh)*(Yvals[i_cnt+1]-thresh) > 0.0:
        i_cnt += 1
    x1= Xlinterp(Yvals[i_cnt], Yvals[i_cnt+1], Xvals[i_cnt], Xvals[i_cnt+1], thresh)

    i_cnt= len(Yvals)-1
    while (Yvals[i_cnt]-thresh)*(Yvals[i_cnt-1]-thresh) > 0.0:
        i_cnt -=1
    x2= Xlinterp(Yvals[i_cnt], Yvals[i_cnt-1], Xvals[i_cnt], Xvals[i_cnt-1], thresh)

    return (x2+x1)/2.0, abs(x2-x1)


def FindNearest(array, value):
    """
    Return the index of the number in array that is closest to value
    stolen from stackoverflow.com from unutbu
    """
    if value-max(np.diff(array)) > max(array) or value+max(np.diff(array)) < min(array):
        raise Exception("value outside of array range")
    if type(array) == list:
        array= np.array(array)
    idx=(np.abs(array-value)).argmin()
    return idx


def CompareTwoLists(Lst1, Lst2, Operation='AND', FailOnError=False):
    """Operation='AND', return a list with common values
       Operation='OR', return a list containing all unique values but no repeats
       If FailOnError=True, raise a ValueError exception rather than return an empty list
       Inputs can be arrays instead of lists, but lists are always returned
    """
    err_msg= None
    if Operation.upper() == 'AND':
        nlst= list(set(Lst1) & set(Lst2))
    elif Operation.upper() == 'OR':
        nlst= list(set(Lst1) | set(Lst2))
    else:
        err_msg= "Error in CompareTwoLists(): Incorrect Operation specified"

    if len(nlst) == 0:
        err_msg= "Error in CompareTwoLists(): Result of operation is empty list"
    if FailOnError and err_msg != None:
        raise ValueError(err_msg)
    else:
        return nlst


def GetZeroCrossings(Vals, Rising=True):
    """
       Assumes equally spaced values on x-axis and returns float zero-crossing values relative to indices by performing linear interpolation
       
       If Rising is True, return only the zero crossings with positive slopes.
       If Rising is False, return only the zero crossings with negative slopes.
       Vals can be list or ndarray
       (Adapted from code on https://gist.github.com/endolith/255291)
    """
    vals= Vals
    if type(Vals) == list:
        vals= np.array(Vals)
    if Rising:
        indices = find((vals[1:] >= 0) & (vals[:-1] < 0))   #indices right before RISING-EDGE zero crossings only
        nvals= vals[indices]
        nvals_p1= vals[indices+1]
        crossings = 1.0*indices - 1.0*nvals/(nvals_p1-nvals)
    else:
        crossings= GetZeroCrossings(-1.0*vals, Rising=True)  #use recursion
    return crossings
    
    
def tabPrint(strng, tab_tuple):
    """TODO: this needs to be fixed"""
    if type(tab_tuple) == bool:
        if tab_tuple:    print(string)

def find_all(a_string, sub):
    '''Yields all the positions of the pattern p in the string s.
       Stolen from https://stackoverflow.com/questions/4664850/how-to-find-all-occurrences-of-a-substring
    '''
    result = []
    k = 0
    while k < len(a_string):
        k = a_string.find(sub, k)
        if k == -1:
            return result
        else:
            result.append(k)
            k += 1 #change to k += len(sub) to not search overlapping results
    return result

def CleanString(in_string, good_chars=['0','1','2','3','4','5','6','7','8','9',',','.','-'], rep_char='', verbose=False):
    """
    clean spurious chars from a string containing comma delimitted values.  only keep chars listed in good_chars
    if rep_char != '' replace the bad chars with the provided string
    there's probably a faster way to do this, but time used isn't too bad so far
    """
    newstr= ''
    bad_chars= ''
    bad_count= 0
    for char in in_string:
        if char in good_chars:
            newstr += char
        else:
            bad_char= "%s"%repr(char)
            if bad_char not in bad_chars:
                bad_chars += bad_char
            bad_count += 1
            newstr += rep_char
    if len(bad_chars) > 0:
        msg= 'which were deleted'
        if rep_char != '':
            msg= "which were replaced with '%s'"%rep_char
        if verbose:  print('WARNING: CleanString() found %d of the following bad chars %s: %s'%(bad_count,msg,bad_chars))
    return newstr
            

def print_nolf(strng, add_comma=True):
    """print string without a linefeed & carraige return"""
    comma_str= ''
    if add_comma:
        comma_str= ', '
    print(repr(strng)[1:-1]+comma_str, end='', flush=True)


def print_bytes(byte_array, ints_per_line=10, header=True):
    """show the integers in a byte array in a nice format"""
    if header:
        for jj in range(ints_per_line):
            print_nolf("  %3d"%jj, False)
        print("\r")
        print("-"*ints_per_line*5)    
    dlst= list(byte_array)    #converts bytes to ints
    line_cnt= int(len(dlst)/ints_per_line)
    extra= len(dlst)%ints_per_line
    for ii in range(line_cnt):
        for jj in range(ints_per_line):
            print_nolf("  %03d"%dlst[ints_per_line*ii+jj], False)
        print("\r")
    for kk in range(extra):
        print_nolf("  %03d"%dlst[ints_per_line*(ii+1)+kk], False)


def search_sequence_numpy(arr,seq):
    """ Find sequence in an array using NumPy only.
    Parameters
    ----------    
    arr    : input 1D array
    seq    : input 1D array
    Output
    ------    
    Output : 1D Array of indices in the input array that satisfy the 
    matching of input sequence in the input array.
    In case of no match, an empty list is returned.
    
    Stolen from: https://stackoverflow.com/questions/36522220/searching-a-sequence-in-a-numpy-array
    """
    # Store sizes of input array and sequence
    Na, Nseq = arr.size, seq.size

    # Range of sequence
    r_seq = np.arange(Nseq)

    # Create a 2D array of sliding indices across the entire length of input array.
    # Match up with the input sequence & get the matching starting indices.
    M = (arr[np.arange(Na-Nseq+1)[:,None] + r_seq] == seq).all(1)

    # Get the range of those indices as final output
    if M.any() >0:
        return np.where(np.convolve(M,np.ones((Nseq),dtype=int))>0)[0]
    else:
        return []            


def argmax_2d(vals):
    """
    Ref: https://stackoverflow.com/questions/3584243/get-the-position-of-the-largest-value-in-a-multi-dimensional-numpy-array
    """
    dum= np.argmax(vals)
    return np.unravel_index(dum, vals.shape)

def argmin_2d(vals):
    """
    Ref: https://stackoverflow.com/questions/3584243/get-the-position-of-the-largest-value-in-a-multi-dimensional-numpy-array
    """
    dum= np.argmin(vals)
    return np.unravel_index(dum, vals.shape)

def stairstep(xs, ys):
    """
    return new x,y arrays that when plotted show a stair step version of the curve represented by the original arrays. 
    no assumptions are made regarding constant step size
    """
    nxs= np.zeros( 2*len(xs)-1, np.float)
    nys= np.zeros( 2*len(xs)-1, np.float)
    nxs[::2]= xs
    nys[::2]= ys
    nxs[1::2]= xs[1:]
    nys[1::2]= ys[:-1]
    return nxs, nys


def transform_vectors_2d(inis, fins=None, dx=0, dy=0, theta_deg=0):
    """
    rotation of points (by theta_deg) and translation (by dx and dy) transformation within a fixed coord system (this is not rotation of coord system)
    if fins=None just transform the points in inis
    return the coords of points in new coord system that is shifted by (dx,dy) and rotated by theta_deg
    """
    from ray3d import get_rot_matrix
    #get affine 2D translation matrix (regular translation matrix is not linear in cartesian coords...must use affine version)
    nn= np.ones( (3,3), np.float )
    nn[1,0],nn[0,1]= (0,0)
    nn[2,0],nn[2,1]= (0,0)
    nn[0,2]= dx
    nn[1,2]= dy
    #get affine rotation matrix
    mm= get_rot_matrix(theta_deg*np.pi/180.0, 'z')
    #mm= mm[:2,:2]   #do NOT reduce to 2D (see https://en.wikipedia.org/wiki/Transformation_matrix regarding affine transformations)
    #get net transformation matrix
    pp= np.matmul(nn,mm)    #order matters!
    #need to add a column of ones to vectors for affine trans
    size= np.shape(inis)[0]
    ninis= np.hstack( (inis,np.ones( (size,1), np.float)) )
    if fins is not None:
        nfins= np.hstack( (fins,np.ones( (size,1), np.float)) )
    #finally can transform the input rays
    ninis= np.matmul(pp, ninis.transpose()).transpose()
    if fins is not None:
        nfins= np.matmul(pp, nfins.transpose()).transpose()
        return ninis[:,:-1], nfins[:,:-1]   #need to slice off extra dimension that was there just for the math
    else:
        return ninis[:,:-1]
    

def get_rot_matrix(theta_rad, axis='x'):
    """
    return matrix for rotating a point about an axis in a fixed coord system
    Ref: https://www.meccanismocomplesso.org/en/3d-rotations-and-euler-angles-in-python/
    returns type np.ndarray so you have to use np.matmul() to multiply rotation matrix by location array using matrix multiplication rules
    """
    import math as m
    theta= theta_rad
    if axis == 'x':
        return np.array([[ 1, 0           , 0           ],
                   [ 0, m.cos(theta),-m.sin(theta)],
                   [ 0, m.sin(theta), m.cos(theta)]])
    elif axis == 'y':
        return np.array([[ m.cos(theta), 0, m.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-m.sin(theta), 0, m.cos(theta)]])
    elif axis == 'z':
        return np.array([[ m.cos(theta), -m.sin(theta), 0 ],
                   [ m.sin(theta), m.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])

    
def transform_points_3d(vals, axis='x', dx=0, dy=0, dz=0, theta_deg=0, trans_mat=None, mat_only=False):
    """
    rotation and translation transformation on vals array, shape=(N,3) which are points in 3D cartesian space
    if trans_mat is not None then just apply this transformation (typically the inverse of a previous trans)
    if mat_only is True then just return the calculated transformation (vals are ignored)
    axis specifies the rotation axis only
    """
    pp= trans_mat
    if pp is None:
        #get affine 3D TRANSLATION matrix (regular translation matrix is not linear in cartesian coords...must use affine version)
        nn= np.diag( [1.0,1.0,1.0,1.0] )   #create 4x4 identity array of floats
        nn[0,3]= dx   
        nn[1,3]= dy
        nn[2,3]= dz
        #get 3D affine ROTATION matrix 
        mm= get_rot_matrix(theta_deg*np.pi/180.0, axis)
        mm= np.vstack( (mm,np.zeros(3)) )
        mm= np.vstack( (mm.transpose(),np.zeros(4)) ).transpose()
        mm[3,3]= 1.0
        #get net transformation matrix
        pp= np.matmul(nn,mm)   #not sure about commutativity here but this order works
    if mat_only:
        return pp
    #need to add a column of ones to vectors for affine trans
    size= np.shape(vals)[0]
    nvals= np.hstack( (vals,np.ones( (size,1), np.float)) )
    #finally can transform the input rays
    nvals= np.matmul(pp, nvals.transpose()).transpose()
    return nvals[:,:-1]    

def find_max_y(xs, ys, quad_fit=False, verbose=True, fignum=None):
    imax= np.argmax(ys)
    if quad_fit:
        ii= min(2, imax, len(ys)-imax-1)  #typically will do quadratic fit to 5 values (or at least 3)
        if ii == 0:
            tabPrint("*****WARNING: Can't do quad fit.  Array too small or extreme value at end point*****", verbose)
            return xs[imax], ys[imax]            
        dum= FitLineToPlot(xs[imax-ii:imax+ii+1], ys[imax-ii:imax+ii+1], 2, Label='theta', Fignum=fignum)
        tabPrint("Using quadratic fit to get best theta value...Rsq=%5.3f"%dum.Rsq, verbose)
        x_max= -dum.Coeffs[1]/(2*dum.Coeffs[0])
        y_max= dum.Coeffs[0]*x_max**2 + dum.Coeffs[1]*x_max + dum.Coeffs[2]
        return x_max, y_max
    else:
        return xs[imax], ys[imax]

    
def find_min_y(xs, ys, quad_fit=False, verbose=True, fignum=None):
    nys= -1*np.array(ys)
    return find_max_y(xs, nys, quad_fit=quad_fit, verbose=verbose, fignum=fignum)


def plot_text_box(text_str, ax=0, loc_pcts=(30,10)):
    """loc_pcts is a tuple specifying the lower left hand corner location of the text box on the plot.  Set to None to ignore
       set fignum=ax"""
    text_str= str(text_str)
    if ax is None or ax == 0:
        fig= plt.figure()
        ax= fig.add_subplot(111)
        ax.plot(np.arange(5), np.arange(5), color='white')  #make invisible line just to set some min and max coords
    elif type(ax) is int:
        fig= plt.figure(ax)
        ax= fig.add_subplot(111)        
    xmin, xmax= ax.set_xlim()
    ymin, ymax= ax.set_ylim()
    ax.text(xmin+0.01*loc_pcts[0]*(xmax-xmin), ymin+0.01*loc_pcts[1]*(ymax-ymin), text_str, color='black', fontsize=8, \
                     bbox=dict(facecolor='blue', alpha=0.3))


def plot_plane(point, normal, width=10, color='green', ret_vals=False, ax=0):
    d = -point.dot(normal)

    # create x,y
    xs, ys = np.meshgrid(np.arange(point[0]-width, point[0]+width), np.arange(point[1]-width, point[1]+width))

    # calculate corresponding z
    zs = (-normal[0] * xs - normal[1] * ys - d) * 1. /normal[2]
    
    # plot the surface
    if ax == 0:
        ax= GetAx(three_d=True)
    ax.plot_surface(xs, ys, zs, alpha=0.2, color=color)
    ax.scatter([point[0]], [point[1]], [point[2]], 'o', s=10, color=color)
    if ret_vals:
        return xs, ys, zs
    
