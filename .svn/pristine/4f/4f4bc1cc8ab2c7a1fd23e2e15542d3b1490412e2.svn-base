'''
Created on 17 nov. 2016

@author: ale
'''
import numpy as np
import random

class Neurona():
    '''
    classdocs
    '''
    def __init__(self, Nentrada, PesoInicial, virtual):
        #Virtual define si la neurona tendra entrada virtual
        self.Ne = Nentrada
        self.salida = 0
        self.delta = 0
        self.aCero = -1
        self.wCero = 0
        self.virtual = virtual
        
        self.pesos = np.zeros(self.Ne)
        
        #Poblar pesos
        if(hasattr(PesoInicial, "__len__")):
            i=0
            while i<self.Ne:
                a = PesoInicial[1]
                self.pesos[i] = random.uniform(-a, a)
                i+=1
                if(self.virtual==1):
                    self.wCero = random.uniform(a, a)
        else:
            i = 0
            while i<self.Ne:
                self.pesos[i] = PesoInicial
                i+=1
            if(virtual == 1):
                self.wCero = PesoInicial
                
        self.pesoAnterior = np.zeros(self.Ne)