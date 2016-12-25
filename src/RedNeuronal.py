import math
import numpy as np
from Neurona import Neurona

def sigmoide(x):
    a = (1+math.exp(-x))
    return 1/a

    #return math.tanh(x)
    
def dsigmoide(x):
    return (x*(1-x))
    #return 1.0-x**2

class RedNeuronal():
    
    #Constructor
    def __init__(self, Nentrada, Noculta, Nsalida, Ncapas, PesoInicial, factorAprendizaje, clasificacion, virtual):
        #Clasificación es un boolean(1 para sí)
        #Virtual boolean para definir si hay entrada virtual en las neuronas
        
        self.Ne = Nentrada
        self.No = Noculta
        self.Ns = Nsalida
        self.Nc = Ncapas
        self.clasificacion = clasificacion
        self.aprendizaje = factorAprendizaje
        self.virtual = virtual
        self.neuronas = []
        
        if not(hasattr(PesoInicial, "__len__")):
            a = PesoInicial
            PesoInicial = []
            PesoInicial.append(a)
            PesoInicial.append(a)
        
        #Crear array vacio(capa nueva)
        array = []
        i = 0
        while i<self.Ne:
            #Introducir neurona nueva en el array vacio
            array.append(Neurona(0, PesoInicial[0], virtual))
            i+=1
        #Incluir array(una capa nueva) a la matriz de neuronas
        self.neuronas.append(array)
            
        #Inicializar primera capa oculta
        #Crear array vacio(capa nueva)
        array = []
        i = 0
        while i<self.No:
            #Introducir neurona nueva en el array vacio
            array.append(Neurona(Nentrada, PesoInicial[1], virtual))
            i += 1
        #Incluir array(una capa nueva) a la matriz de neuronas
        self.neuronas.append(array)
            
        #Inicializar capa oculta
        i=0
        j=2
        #Repetir (sin contar las dos capas ya creadas) hasta conseguir el numero de capas requerido
        while j<self.Nc-1:
            #Crear array vacio(capa nueva)(una vez por cada capa que queremos)
            array = []
            while i<self.No:
                #Introducir neurona nueva en el array vacio
                array.append(Neurona(self.No, PesoInicial[1], virtual))
                i+=1
            #Incluir array(una capa nueva) a la matriz de neuronas
            self.neuronas.append(array)
            i=0
            j+=1
            
        #Inicializar salida
        i=0
        #Crear array vacio(capa nueva)(una vez por cada capa que queremos)
        array = []
        while i<self.Ns:
        #Introducir neurona nueva en el array vacio
            array.append(Neurona(self.No, PesoInicial[1], virtual))
            i+=1
        #Incluir array(una capa nueva) a la matriz de neuronas
        self.neuronas.append(array)
            
    #Hacia delante
    def delante(self, entrada):
        
        #Esto deberia ser mas compacto(repasar)
        c = 0
        while c<self.Nc:
            i=0
            if(c==0):
                #Por cada neurona i de la capa(esto para la 1ra capa-> comprobar si c es 0)
                while i<len(self.neuronas[c]):
                    neuActual = self.neuronas[c][i]
                    neuActual.salida = entrada[i]
                    i+=1
                c+=1
            else:
                    #Por cada neurona de la capa actual
                    while i<len(self.neuronas[c]):
                        neuActual = self.neuronas[c][i]
                        n=0
                        j=0
#                         Por cada neurona de la capa anterior
                        while j<len(self.neuronas[c-1]):
                            neuEntrada=self.neuronas[c-1][j]
                            n+= neuActual.pesos[j] * neuEntrada.salida
                            j+=1
                        if(self.virtual == 1):
                            n += neuActual.wCero * neuActual.aCero
                        neuActual.salida = sigmoide(n)
                        i+=1
                    c+=1
    
    def decodifica(self, entrada):
        self.delante(entrada)
        salida = []
        i=0
        while i<self.Ns:
            sal = self.neuronas[self.Nc-1][i].salida
            if(self.clasificacion==1):
                if(sal>0.5):
                    sal = 1
                else:
                    sal = 0
            salida.append(sal)
            i+=1
        
        salida = np.array(salida)
        
        return salida
        
    def atras(self, salidaEsperada):
        #Propaga hacia atrás actualizando el delta de las neuronas
        #Esto deberia ser mas compacto(repasar)
        c = self.Nc-1;
        while c>=0:
            i=0
            if(c==self.Nc-1):
                #Por cada neurona i de la ultima capa
                while i<len(self.neuronas[c]):
                    neuActual = self.neuronas[c][i]
                    error = salidaEsperada[i] - neuActual.salida
                    neuActual.delta = dsigmoide(neuActual.salida) * error
                    i+=1
                c-=1
            else:
                    #Por cada neurona de la capa actual
                    while i<len(self.neuronas[c]):
                        neuActual = self.neuronas[c][i]
                        error=0
                        j=0
#                         Por cada neurona de la capa siguiente
                        while j<len(self.neuronas[c+1]):
                            neuSiguiente=self.neuronas[c+1][j]
                            error+= neuSiguiente.delta * neuSiguiente.pesos[i]
                            j+=1
                        neuActual.delta = dsigmoide(neuActual.salida) * error
                        i+=1
                    c-=1  
        
    def actualizaPesos(self):
        #Esto deberia ser mas compacto(repasar)
        c = 1
        while c<self.Nc:
            i=0
            #Por cada neurona de la capa actual
            while i<len(self.neuronas[c]):
                neuActual = self.neuronas[c][i]
                j=0
                #Por cada neurona de la capa anterior
                while j<len(self.neuronas[c-1]):
                    pesoActual = neuActual.pesos[j]
                    neuEntrada=self.neuronas[c-1][j]
                    pesoActualizado = pesoActual + self.aprendizaje * neuEntrada.salida * neuActual.delta
                    
                    neuActual.pesoAnterior[j] = pesoActual
                    neuActual.pesos[j] = pesoActualizado
                    j+=1
                if(self.virtual==1):
                    pesoActual = neuActual.wCero
                    entrada = neuActual.aCero
                    pesoActualizado = pesoActual + self.aprendizaje * entrada * neuActual.delta
                    neuActual.wCero = pesoActualizado
                i+=1
            c+=1
    
    def cargaPesos(self, pesos):
        i=self.Ne
        numTotalNeuronas = self.Ne + (self.Nc-2)*self.No + self.Ns
        while i<numTotalNeuronas:
            neurona = self.obtenNeurona(i)
            pi = np.copy(pesos[i-self.Ne])
            neurona.wCero = pi[neurona.Ne]
            pi.resize(neurona.Ne)
            neurona.pesos = pi
            i+=1
            
    def devuelvePesos(self):
        maxElementos = (self.Nc-2)*self.No+self.Ns
        maxEntradas = max(self.Ne, self.No)+1
        pesos = np.zeros(shape=[maxElementos, maxEntradas])
        
        numTotalNeuronas = maxElementos
        for i in range(numTotalNeuronas):
            j = i+self.Ne
            neurona = self.obtenNeurona(j)
            pi = np.append(neurona.pesos, neurona.wCero)
            pi.resize(maxEntradas)
            pesos[i] =  pi
            
        return pesos
            
    
    def obtenNeurona(self, i):
        #Obtiene una neurona dado un indice (empezando en 0)     
        NumTotalNeuronas = self.Ne + (self.Nc-2)*self.No + self.Ns
        
        #Comprobar si es una neurona de entrada
        if(i<self.Ne):
            Ncapa=0
            Nneurona = i
            neurona = self.neuronas[Ncapa][Nneurona]
        
        #Comprobar que es oculta:
        elif(i<(NumTotalNeuronas-self.Ns)):
            if(i>=self.Ne):
                Ncapa = ((i-self.Ne)//self.No)+1
                Nneurona = i - ((Ncapa-1)*self.No) - self.Ne
                neurona = self.neuronas[Ncapa][Nneurona]
        
        #Comprobar si es de salida
        
        else:
            Nocultas = (self.Nc-2)*self.No
            Ncapa = self.Nc-1
            Nneurona = i-Nocultas-self.Ne
            neurona = self.neuronas[self.Nc-1][Nneurona]
        return neurona
    
    def aprende(self, entrada, salidaEsperada, numRepeticiones):
        #Dada una entrada y una salida esperada, calcula la salida y actualiza los pesos
        i=0
        while i<numRepeticiones:
            self.delante(entrada)
            self.atras(salidaEsperada)
            self.actualizaPesos()
            i+=1
        
    def aprendeLote(self, entradas, salidasEsperadas, numRepeticiones):
        #Dado un lote de entradas y salidas esperadas, calcula la salida y actualiza los pesos para cada una. Nº de repeticiones para cada entrada opcional
        
        for i in range(numRepeticiones):
            a = np.arange(len(entradas))
            np.random.shuffle(a)
            for i in a:
                self.aprende(entradas[i], salidasEsperadas[i], 1)
