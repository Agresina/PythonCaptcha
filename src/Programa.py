from RedNeuronal import RedNeuronal as redNeural
import numpy as np
import utils


def testEjercicioRed():
    Nentrada = 2
    Noculta = 2
    Nsalida = 2
    Ncapas = 4
    PesoInicial= 1
    factorAprendizaje = 0.1
    
    entradas = [1, 1]
    salidaEsperada = [1, 1]
    
    pesoEsperado = 0.994809355204
    
    red = redNeural(Nentrada, Noculta, Nsalida, Ncapas, PesoInicial, factorAprendizaje, 0, 1)
    
    red.aprende(entradas, salidaEsperada, 1)
    
    print("Prueba sobre el ejercicio 3 de la relacion")
    print("Tras una iteración del método de retropropagación, el peso W05 (La sexta neurona, empezamos desde 0) es:")
    print(red.obtenNeurona(5).wCero)
    print("El peso esperado es: ")
    print(pesoEsperado)
    print("")

def testEjercicioRed2():
    Nentrada = 2
    Noculta = 2
    Nsalida = 2
    Ncapas = 4
    PesoInicial= 1
    factorAprendizaje = 0.1
    
    entradas = [0.5, 0.5]
    salidaEsperada = [0.8, 0.8]
    
    pesoEsperado = 1.001875
    
    red = redNeural(Nentrada, Noculta, Nsalida, Ncapas, PesoInicial, factorAprendizaje, 0, 1)
    
    red.aprende(entradas, salidaEsperada, 1)
    print("Prueba sobre el ejercicio 12 de la relacion")
    print("Tras una iteración del método de retropropagación, el peso W36 (De la neurona 3 a la neurona 6 es:")
    print(red.obtenNeurona(5).pesos[0])
    print("El peso esperado es: ")  
    print(pesoEsperado)
    print("")


def testLetras():
    #Estructura de la red neuronal:
    Nentrada = 900
    #Al aumentarlo ha aumentado la diferencia entre lasneuronas, pero la salida es muy baja
    Noculta = 50
    Nsalida = 6
    Ncapas = 3
    #Al aumentarlo ha aumentado la salida bastante, apenas hay diferencia porque se queda en 0.99.
    # En 0.01 es muy bajo 0.15-0.26
    # EN 0.03 0.11-0.24
    PesoInicial= [["rand", 0.1], ["rand", 0.5]]
    factorAprendizaje = 0.1

    print("Test de aprendizaje sobre letras 1. Repeticiones de aprendizaje:100")
    red = redNeural(Nentrada, Noculta, Nsalida, Ncapas, PesoInicial, factorAprendizaje, 1, 1)
    
    o = np.array([[1,0,0,0,0,0]] * 20)
    p = np.array([[0,1,0,0,0,0]] * 20)
    q = np.array([[0,0,1,0,0,0]] * 20)
    r = np.array([[0,0,0,1,0,0]] * 20)
    s = np.array([[0,0,0,0,1,0]] * 20)
    t = np.array([[0,0,0,0,0,1]] * 20)
    
    o2 = np.array([[1,0,0,0,0,0]] * 5)
    p2 = np.array([[0,1,0,0,0,0]] * 5)
    q2 = np.array([[0,0,1,0,0,0]] * 5)
    r2 = np.array([[0,0,0,1,0,0]] * 5)
    s2 = np.array([[0,0,0,0,1,0]] * 5) 
    t2 = np.array([[0,0,0,0,0,1]] * 5)
    
    esp = np.concatenate([o,p,q,r,s,t])
    esp2 = np.concatenate([o2,p2,q2,r2,s2,t2])
    
    TRAIN_SET_DIRECTORY = './data/*.pgm'
    TEST_SET_DIRECTORY = './test/*.pgm'
    
    letras = utils.lettersToArray(TRAIN_SET_DIRECTORY)
    letrasTest = utils.lettersToArray(TEST_SET_DIRECTORY)
    
    print("Aprendiendo...")
    red.aprendeLote(letras, esp, 100)
    
    print("Aprendizaje completo, guardando resultado...")
    utils.writePesos("pesos.txt", red.devuelvePesos())
    
    print("Prueba de lectura de una letra ([1 0 0 0 0 0]")
    print(red.decodifica(letras[0]))

    print("Evaluación de rendimiento")
    exito=0
    fallo=0
    for i in range(30):
        print("la letra es: ")
        salida = red.decodifica(letrasTest[i])
        testDec = checkRespuesta(salida, esp2[i])
        if(testDec):
            exito+=1
        else:
            fallo+=1
    
    print("Tasa de éxito: ")
    print(100*exito/(exito+fallo))
    print("Exito: " + str(exito))
    print("Fallo: " + str(fallo))
    print("")
def testLetras2():
    #Estructura de la red neuronal:
    Nentrada = 900
    #Al aumentarlo ha aumentado la diferencia entre lasneuronas, pero la salida es muy baja
    Noculta = 50
    Nsalida = 6
    Ncapas = 3
    #Al aumentarlo ha aumentado la salida bastante, apenas hay diferencia porque se queda en 0.99.
    # En 0.01 es muy bajo 0.15-0.26
    # EN 0.03 0.11-0.24
    PesoInicial= [["rand", 0.1], ["rand", 0.5]]
    factorAprendizaje = 0.1

    print("Test de aprendizaje sobre letras 2. Cargar red neuronal ya entrenada")
    red = redNeural(Nentrada, Noculta, Nsalida, Ncapas, PesoInicial, factorAprendizaje, 1, 1)
    
    #o = np.array([[1,0,0,0,0,0]] * 20)
    #p = np.array([[0,1,0,0,0,0]] * 20)
    #q = np.array([[0,0,1,0,0,0]] * 20)
    #r = np.array([[0,0,0,1,0,0]] * 20)
    #s = np.array([[0,0,0,0,1,0]] * 20)
    #t = np.array([[0,0,0,0,0,1]] * 20)
    
    o2 = np.array([[1,0,0,0,0,0]] * 5)
    p2 = np.array([[0,1,0,0,0,0]] * 5)
    q2 = np.array([[0,0,1,0,0,0]] * 5)
    r2 = np.array([[0,0,0,1,0,0]] * 5)
    s2 = np.array([[0,0,0,0,1,0]] * 5) 
    t2 = np.array([[0,0,0,0,0,1]] * 5)
    
    #esp = np.concatenate([o,p,q,r,s,t])
    esp2 = np.concatenate([o2,p2,q2,r2,s2,t2])
    
    TRAIN_SET_DIRECTORY = './data/*.pgm'
    TEST_SET_DIRECTORY = './test/*.pgm'
    
    letras = utils.lettersToArray(TRAIN_SET_DIRECTORY)
    letrasTest = utils.lettersToArray(TEST_SET_DIRECTORY)
    
    #print("Aprendiendo...")
    #red.aprendeLote(letras, esp, 100)
    
    #print("Aprendizaje completo, guardando resultado...")
    #utils.writePesos("pesos.txt", red.devuelvePesos())
    print("Cargando datos...")
    red.cargaPesos(utils.readPesos("pesosNo50Nc3LetrasO-Rep100.txt"))
    
    print("Prueba de lectura de una letra ([1 0 0 0 0 0])")
    print(red.decodifica(letras[0]))

    print("Evaluación de rendimiento")
    exito=0
    fallo=0
    for i in range(30):
        print("la letra es: ")
        salida = red.decodifica(letrasTest[i])
        testDec = checkRespuesta(salida, esp2[i])
        if(testDec):
            exito+=1
        else:
            fallo+=1
    
    print("Tasa de éxito: ")
    print(100*exito/(exito+fallo))
    print("Exito: " + str(exito))
    print("Fallo: " + str(fallo))
    print("")

def checkRespuesta(salida, esperada):
    print("La salida es" + str(salida))
    print("Lo esperado es " + str(esperada))
    return np.array_equal(salida, esperada)

testEjercicioRed()
testEjercicioRed2()
testLetras()
testLetras2()