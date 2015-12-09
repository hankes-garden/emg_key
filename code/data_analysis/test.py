# -*- coding: utf-8 -*-
"""
Created on Mon Dec 07 14:46:23 2015

@author: jason
"""

import matlab.engine

eng = matlab.engine.start_matlab()
tf = eng.isprime(37)
print(tf)