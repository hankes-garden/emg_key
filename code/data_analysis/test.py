import numpy as np
import reedsolo as RS


def toBinaryArray(arrData):
    arrBits = None
    for i in arrData:
        # [2:] to chop off the "0b" part
        arr = np.array([int(digit) for digit in bin(i)[2:]]) 
        nLen = len(arr)
        if (nLen<3):
            arr = np.concatenate((np.zeros(3-nLen, dtype=np.int), arr))
        arrBits = np.concatenate((arrBits, arr) ) \
            if arrBits is not None else arr
    return arrBits
        
lsSegmentCode = [0, 1, 1, 2, 1, 3, 2, 2, 2, 2, 3, 2, 0, 4, 2, 0, 0, 0, 2, 3, 2, 2, 2, 1, 4, 2, 0, 0]
arrCode = bytearray(lsSegmentCode)

rs = RS.RSCodec(5)
e = rs.encode([1, 2 ,9])
d = rs.decode(e)
for i in e:
    print i


        
    
