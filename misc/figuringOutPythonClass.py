# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 13:36:50 2016

@author: deanpospisil
"""
from . import x
class A(object):
    def __init__(self):
        self.x = 'Hello'

    def method_a(self, foo):
        print(self.x + ' ' + foo)
        
        
a = A()

