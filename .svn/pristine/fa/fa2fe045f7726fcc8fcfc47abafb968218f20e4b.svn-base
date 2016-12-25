import glob2
import numpy as np

def imageToArray(image_path):
    with open(image_path, 'r') as f:
        result = []
        for line in f:
            for word in line.split():
                if word == '255':
                    result.append(1)
                elif word == '0':
                    result.append(0)
        return result[0:900]


# A partir de un directorio devuelve un numpy_array
def lettersToArray(letters_directory_path):
    file_paths = sorted(glob2.glob(letters_directory_path))
    c = 0
    numpy_array = np.zeros(shape=(120, 900), dtype=np.int)
    for file_path in file_paths:
        array = imageToArray(file_path)
        numpy_array[c] = array
        c += 1
    return numpy_array

def readPesos(pesos_path):
    pesos = np.loadtxt(pesos_path)
    return pesos
    
def writePesos(pesos_path, array):    
    np.savetxt(pesos_path,array)
    
