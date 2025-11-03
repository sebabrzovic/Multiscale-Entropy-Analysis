import numpy as np
import numpy.linalg as nl
import random
import matplotlib.pyplot as plt
import networkx as nx

from arithmetic_compressor import AECompressor
from arithmetic_compressor.models import StaticModel

############### ARITHMETIC COMPRESSION ####################

def get_compression_length(binary_string):
    """
    Takes a binary string and returns the length of its arithmetic coding compression
    
    Args:
        binary_string (str): A string of '0's and '1's
        
    Returns:
        int: Length of the compressed data
    """
    # Create model - using 0.5 probability for both as a baseline
    # You might want to adjust these probabilities based on your data
    binary_model = StaticModel({'0': 0.5, '1': 0.5})
    
    # Create compressor
    coder = AECompressor(binary_model)
    
    # Compress and get length
    compressed = coder.compress(binary_string)
    return len(compressed)
  
  
def get_binary_probabilities(binary_string):
    MIN_PROB = 0.0001  # Minimum probability allowed for any symbol
    
    total_len = len(binary_string)
    zeros = binary_string.count('0')
    ones = binary_string.count('1')
    
    # Calculate raw probabilities
    raw_prob_0 = zeros / total_len
    raw_prob_1 = ones / total_len
    
    # Apply minimum probability threshold
    if raw_prob_0 < MIN_PROB:
        prob_0 = MIN_PROB
        prob_1 = 1 - MIN_PROB
    elif raw_prob_1 < MIN_PROB:
        prob_1 = MIN_PROB
        prob_0 = 1 - MIN_PROB
    else:
        # If both probabilities are above minimum, use raw probabilities
        prob_0 = raw_prob_0
        prob_1 = raw_prob_1
        
    return {'0': prob_0, '1': prob_1}

def get_optimized_compression_length(binary_string):
    # Create model with actual probabilities from the data
    probabilities = get_binary_probabilities(binary_string)
    binary_model = StaticModel(probabilities)
    
    coder = AECompressor(binary_model)
    compressed = coder.compress(binary_string)
    return len(compressed)  
  
############### GRAPH ENTROPY ####################

def binarizar(decimal):
    binario = ''
    while decimal // 2 != 0:
        binario = str(decimal % 2) + binario
        decimal = decimal // 2
    return str(decimal) + binario

def borrar(V, k): #V lista de listas, k entero a eliminar
    for i in range(len(V)):
      if (k in V[i]):
        V[i].remove(k)
    return V #Eliminamos elemento k de V

def particionar(matrix, array, k): #k nodo
    particion1 = [] #vecinos
    particion2 = [] #no vecinos
    for j in array:
      if matrix[k-1][j-1] == 1:
        particion1.append(j)
      elif j != k: #Para no guardar al mismo elemento
        particion2.append(j)
    return [particion1,particion2]

def binario_conj(array_conj , array_vec):
    bin_set = binarizar(len(array_conj))
    bin_vec = binarizar(len(array_vec))
    while len(bin_set)> len(bin_vec): #Aumentamos la codificación para poder
                                      #representar a todos los números de vecinos posibles.
      bin_vec = '0'+bin_vec
    return bin_vec

def M_adyacencia(G):
   #G grafo
   node_to_index = {node: i for i, node in enumerate(G.nodes)}
   n=len(G.nodes)
   M = np.zeros([n,n])
   E = np.array(G.edges) #aristas
   print(n)
   print(len(E))
   
   for e in G.edges: # Agregamos 1 para vértices vecinos
      i = node_to_index[e[0]]
      j = node_to_index[e[1]]
      M[i][j]=1
      M[j][i]=1
   return M

## Codificación de grafos usando Szip algorithm ##
def Encoder (M): #Codificando grafos
   #M matriz adyacencia
   n = np.shape(M)[0] #Canditad de vértices
   V = np.linspace(1,n,n)
   x_0 = V[0]
   P_0 = np.delete(V,0) #Se elimina primer elemento
   E_0 = []
   co_E_0 = []
   for i in range (0,n): #Guardamos vecinos de x_0
    if M[0][i] == 1:
      E_0.append(i+1)
    else:         #No vecinos
      if i !=0:
        co_E_0.append(i+1)

   l = len(E_0) #Cantidad de vecinos para pasarlo a n° binario
   bin_set = binarizar(n)
   bin_0 = binarizar(l)
   while len(bin_set)> len(bin_0): #Aumentamos la codificación para poder
                                   #representar a todos los números de vecinos posibles.
    bin_0 = '0'+bin_0

   if E_0 != []:
      P_k= [E_0, co_E_0]
   else:
      P_k= [co_E_0]

   B1 =''
   B2 = ''
   if len(bin_0)>1:
    B1=bin_0
   else:
    B2=bin_0

   while P_k != []:
    #print(str(P_k) + 'p')
    conj=P_k[0] #Analizamos primer conjunto de partición
    #print(str(conj)+'c')
    #while conj !=[]:
    elem=conj[0]
    #print(str(elem)+'elem')
    P_k_sig=[] #Partición a actualizar

    borrar(P_k,elem) #Primero borramos elemento de la
                     #partición para empezar a contar vecinos

    for subconj in P_k:
     # print(str(subconj)+'sc')
      if subconj !=[]:
        particion = particionar(M, subconj, elem) # [vecinos, no vecinos]
        bin_vec = binario_conj(subconj, particion[0])
        #print(particion)

        if len(bin_vec)>1: #Agregamos codificación
          B1=B1+bin_vec
        else:
          B2=B2+bin_vec

        if len(particion[0])>0: #Guardo particiones solo si son no vacias
          P_k_sig.append(particion[0])
        if len(particion[1])>0:
          P_k_sig.append(particion[1])

    P_k = P_k_sig #Actualizamos partición para la siguiente iteración de elemento
    #print(str(P_k)+'act')

   return (B1,B2,n) #Retorna codificación de grafo


def calculo_de_entropia(Gra):
    M= M_adyacencia(Gra)
    cod = Encoder(M)
    B1,B2, n= cod
    return entropia_LZ76(B1,B2,n)
#importar codificaciongrafico

def separar_Cod(B1,B2,n): #B1, B2 binarios
    n_bin=binarizar(n)
    B=B1+B2+str(n_bin)
    L=len(B)
    part_0=[str(B[0])] #guardamos el primer elemento
    k=0
    while k+1 <= (L-1): #empezamos a recorrer B
      k=k+1
      elem=''+str(B[k])

      while elem in part_0: #buscamos elemento que no haya
                            #sido agregado antes
          if k==(L-1):
            part_0.append(elem)
            return part_0

          else:
            k=k+1
            elem=elem+str(B[k])

      part_0.append(elem)

    return part_0 #retorna la codificación separada de tal
                  #manera que no exista repetición entre particiones


#Dos maneras implementadas para calcular entropía

#Forma 1
def calcular_Entropia(particion): #partición part_0
    sum=0
    for x in particion:
        l_x=len(x)
        sum=sum+(1/2)**(l_x)*l_x # Suma p(x)l(x)

    return sum

### ALGORITMO DE CALCULO DE ENTROPÍA USANDO ARITHMETIC ENCODING ###   

def entropiaArithmeticEncoding(Gra, ListaGrafosRandom):
    """
    Calculate entropy using arithmetic encoding for a graph and average entropy for a list of random graphs
    
    Args:
        Gra: Original graph
        ListaGrafosRandom: List of random graphs for comparison
        
    Returns:
        tuple: (compression_length_original, average_compression_length_random)
    """
    # Calculate entropy for original graph
    M = M_adyacencia(Gra)
    cod = Encoder(M)
    B1, B2, n = cod
    B = B1 + B2
    B_arr = []
    for i in B:
        B_arr.append(i)
    

    compression_original = get_optimized_compression_length(B)
    use_total_compression = False
    
    try:
        compressionB1 = get_optimized_compression_length(B1)
        compressionB2 = get_optimized_compression_length(B2)
        compressionB1B2 = compressionB1 + compressionB2
    except ValueError as e:
        print(f"Warning: Error calculating separate B1+B2 compression: {str(e)}")
        print("Using total compression length for all calculations")
        compressionB1B2 = compression_original
        use_total_compression = True
        
    print(f"Entropia de grafo: {compression_original}, Entropía B1 +B2 : {compressionB1B2} y tamaño de la codificación: {len(B_arr)}")
    
    # Calculate average entropy for random graphs
    compression_random_total = 0
    compressionB1B2_random_total = 0
    codification_lengths = []
    
    for idx, G_random in enumerate(ListaGrafosRandom):
        M_r = M_adyacencia(G_random)
        cod_r = Encoder(M_r)
        B1_r, B2_r, n_r = cod_r
        B_r = B1_r + B2_r
        B_arr_r = []
        for i in B_r:
            B_arr_r.append(i)
        
        compression = get_optimized_compression_length(B_r)
        compression_random_total += compression
        
        if not use_total_compression:
            try:
                compressionB1_r = get_optimized_compression_length(B1_r)
                compressionB2_r = get_optimized_compression_length(B2_r)
                compressionB1B2_r = compressionB1_r + compressionB2_r
                compressionB1B2_random_total += compressionB1B2_r
            except ValueError as e:
                print(f"Warning: Error calculating separate B1+B2 compression for random graph {idx}: {str(e)}")
                use_total_compression = True
        
        codification_lengths.append(len(B_arr_r))
    
    # If we had to use total compression, set B1B2 totals equal to the total compression
    if use_total_compression:
        compressionB1B2_random_total = compression_random_total
    
    avg_compression_random = compression_random_total / len(ListaGrafosRandom)
    avg_compressionB1B2_random = compressionB1B2_random_total / len(ListaGrafosRandom)
    avg_codification_length = sum(codification_lengths) / len(codification_lengths)
    
    print(f"Entropia promedio de grafos random: {avg_compression_random:.2f}, Entropia B1+B2 promedio: {avg_compressionB1B2_random:.2f} y tamaño promedio de la codificación: {avg_codification_length:.2f}")
    
    return compression_original, avg_compression_random, compressionB1B2, avg_compressionB1B2_random

def entropiaArithmeticTheoretic(Gra):

    # Calculate entropy for original graph
    M = M_adyacencia(Gra)
    cod = Encoder(M)
    B1, B2, n = cod
    B = B1 + B2
    B_arr = []
    for i in B:
        B_arr.append(i)
    compression_original = get_optimized_compression_length(B)
    print(f"Entropia de grafo: {compression_original} y tamaño de la codificación: {len(B_arr)}")
    
    return compression_original
    

"""
Simple script implementing Kaspar & Schuster's algorithm for
Lempel-Ziv complexity (1976 version).

If you use this script, please cite the following paper containing a sample
use case and further description of the use of LZ in neuroscience:

Dolan D. et al (2018). The Improvisational State of Mind: A Multidisciplinary
Study of an Improvisatory Approach to Classical Music Repertoire Performance.
Front. Psychol. 9:1341. doi: 10.3389/fpsyg.2018.01341

Pedro Mediano and Fernando Rosas, 2019

if __name__ == '__main__':
    # Simple string, low complexity
    ss = np.array([0,0,0,0,0,0,0,1,1,1,1,1,1])
    print('The complexity of the string %s is %i'%(ss, LZ76(ss)))

    # Irregular string, high complexity
    ss = np.array([0,1,1,0,1,0,0,1,0,1,1,1,0])
    print('The complexity of the string %s is %i'%(ss, LZ76(ss)))
"""

