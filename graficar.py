#%%
from turtle import color
import cv2 as cv
from numpy import spacing
import pandas
import matplotlib.pyplot as plt
import os
import pathlib
columnas =['x','y','width','height']


#sort a list
import pathlib
filescsv = [ i for i in os.listdir('.') if i.endswith('.csv')]
f_caminar_3var = sorted([i for i in filescsv if 'F_caminar' in i])
f_correr_3var = sorted([i for i in filescsv if 'F_correr' in i])
f_caminar_xywh = sorted([i for i in filescsv if  i.startswith('caminar')])
f_correr_xywh = sorted([i for i in filescsv if  i.startswith('correr')])
print(filescsv)
#actually work directory
path = str(os.getcwd())
caminar_namesfile_T = [[pandas.read_csv(path+'/' + i),i[:i.find('.')]] for i in f_caminar_3var]
correr_namesfile_T =  [[pandas.read_csv(path+'/' + i),i[:i.find('.')]] for i in f_correr_3var]
caminar_namesfile =   [[pandas.read_csv(path+'/' + i),i[:i.find('.')]] for i in f_caminar_xywh]
correr_namesfile =    [[pandas.read_csv(path+'/' + i),i[:i.find('.')]] for i in f_correr_xywh]


#%%
caminar1 = pandas.read_csv("/home/juanchx/Documentos/prueba deepsort/clasificar/F_caminar_J3.csv")
caminar2 = pandas.read_csv("/home/juanchx/Documentos/prueba deepsort/clasificar/F_caminar_J4.csv")
caminar3 = pandas.read_csv("/home/juanchx/Documentos/prueba deepsort/clasificar/F_caminar_M3.csv")
caminar4 = pandas.read_csv("/home/juanchx/Documentos/prueba deepsort/clasificar/F_caminar_M4.csv")
caminar5 = pandas.read_csv("/home/juanchx/Documentos/prueba deepsort/clasificar/F_caminar_M5.csv")
caminar6 = pandas.read_csv("/home/juanchx/Documentos/prueba deepsort/clasificar/F_caminar_M6.csv")
#caminar_namesfile_T =[[caminar1,'F_caminar_J3'],[caminar2,'F_caminar_J4'],[caminar3,'F_caminar_M3'],[caminar4,'F_caminar_M4'],[caminar5,'F_caminar_M5'],[caminar6,'F_caminar_M6']] 

caminar1o= pandas.read_csv("/home/juanchx/Documentos/prueba deepsort/caminar/caminar_J3.csv")
caminar2o= pandas.read_csv("/home/juanchx/Documentos/prueba deepsort/caminar/caminar_J4.csv")
caminar3o= pandas.read_csv("/home/juanchx/Documentos/prueba deepsort/caminar/caminar_M3.csv")
caminar4o= pandas.read_csv("/home/juanchx/Documentos/prueba deepsort/caminar/caminar_M4.csv")
caminar5o= pandas.read_csv("/home/juanchx/Documentos/prueba deepsort/caminar/caminar_M5.csv")
caminar6o= pandas.read_csv("/home/juanchx/Documentos/prueba deepsort/caminar/caminar_M6.csv")
#caminar_namesfile = [[caminar1o,'caminar_J3'],[caminar2o,'caminar_J4'],[caminar3o,'caminar_M3'],[caminar4o,'caminar_M4'],[caminar5o,'caminar_M5'],[caminar6o,'caminar_M6']]

correr1 = pandas.read_csv("/home/juanchx/Documentos/prueba deepsort/clasificar/F_correr_casa_K.csv")
correr2 = pandas.read_csv("/home/juanchx/Documentos/prueba deepsort/clasificar/F_correr_casa_K2.csv")
correr3 = pandas.read_csv("/home/juanchx/Documentos/prueba deepsort/clasificar/F_correr_casa_K3.csv")
correr4 = pandas.read_csv("/home/juanchx/Documentos/prueba deepsort/clasificar/F_correr_casa_K4.csv")
correr5 = pandas.read_csv("/home/juanchx/Documentos/prueba deepsort/clasificar/F_correr_hacienda_K.csv")
correr6 = pandas.read_csv("/home/juanchx/Documentos/prueba deepsort/clasificar/F_correr_hacienda_K2.csv")
#correr_namesfile_T = [[correr1,'F_correr_casa_K'],[correr2,'F_correr_casa_K2'],[correr3,'F_correr_casa_K3'],[correr4,'F_correr_casa_K4'],[correr5,'F_correr_hacienda_K'],[correr6,'F_correr_hacienda_K2']]

correr1o= pandas.read_csv("/home/juanchx/Documentos/prueba deepsort/correr/casa_K.csv")
correr2o= pandas.read_csv("/home/juanchx/Documentos/prueba deepsort/correr/casa_K2.csv")
correr3o= pandas.read_csv("/home/juanchx/Documentos/prueba deepsort/correr/casa_K3.csv")
correr4o= pandas.read_csv("/home/juanchx/Documentos/prueba deepsort/correr/casa_K4.csv")
correr5o= pandas.read_csv("/home/juanchx/Documentos/prueba deepsort/correr/hacienda_K.csv")
correr6o= pandas.read_csv("/home/juanchx/Documentos/prueba deepsort/correr/hacienda_K2.csv")
#correr_namesfile= [[correr1o,'correr_casa_K'],[correr2o,'correr_casa_K2'],[correr3o,'correr_casa_K3'],[correr4o,'correr_casa_K4'],[correr5o,'correr_hacienda_K'],[correr6o,'correr_hacienda_K2']]


#%%
#redimencionar el dataset # 5 frames por segundo
def redimensionar(dataset_caminar,dataset_correr,reduccion):
    list=[i for i in range(len(dataset_caminar[0][0].iloc[:,0])) if i%reduccion==0]
    max= (int(len(list)*0.15),int(len(list)*0.85))
    for i in range(len(dataset_caminar)):
        #print(len(dataset_caminar[0][0].iloc[:,0]))
        
        #print(dataset_caminar[i][0].iloc[ list , : ]) 
        dataset_caminar[i][0] = dataset_caminar[i][0].iloc[list , : ]
        dataset_caminar[i][0] = dataset_caminar[i][0].reset_index(drop=True)

        dataset_correr[i][0] = dataset_correr[i][0].iloc[list , : ]
        dataset_correr[i][0] = dataset_correr[i][0].reset_index(drop=True)
       
    return dataset_caminar,dataset_correr,max


#redimensionar(caminar_namesfile,correr_namesfile)


#%%
#obtain the len of the dataframe
def graficar(archivo,titulo): 
    fig, axs = plt.subplots(2, 2)
    #size of the graph
    #plt.rcParams["figure.figsize"] = (14,1)
    
    #general title
    fig.suptitle(titulo, fontsize=20, fontweight='bold')
    #space between title and subplots
    fig.subplots_adjust(top=0.85)

    #titles of the graph
    axs[0, 0].set_title(' X vs frames', fontsize=10,color='r')
    axs[0, 1].set_title(' Y vs frames', fontsize=10, color='r')
    axs[1, 0].set_title('Width vs frames',fontsize=10, color='r')
    axs[1, 1].set_title('Height vs frames',fontsize=10, color='r')

    #plot the data
    axs[0, 0].plot(range(len(archivo)), archivo['x'], 'r')
    axs[0, 1].plot(range(len(archivo)), archivo['y'], 'b')
    axs[1, 0].plot(range(len(archivo)),archivo['width'], 'g')
    axs[1, 1].plot(range(len(archivo)),archivo['height'], 'y')
    #ejes x of the first subplot
    axs[0, 0].set_xlabel('Frames')
    axs[0, 1].set_xlabel('Frames')
    axs[1, 0].set_xlabel('Frames')
    axs[1, 1].set_xlabel('Frames')
    axs[0,0].set_ylabel('X')
    axs[0,1].set_ylabel('Y')
    axs[1,0].set_ylabel('whidth')
    axs[1,1].set_ylabel('Height')
    #distance beween the subplots
    fig.subplots_adjust(hspace=0.8, wspace=0.5)


# %%
def espejo_bool(correr,caminar):
    orden_correr=[]
    orden_caminar=[]
    for i in range(len(correr)):
    #print(correr[i][0].iloc[0,:]['x'],)
        for ind,j in enumerate(columnas):
            if correr[i][0].iloc[max[0],:][j] > correr[i][0].iloc[max[1],:][j]:
                orden_correr.append(1)
            else:
                orden_correr.append(0)

            if caminar[i][0].iloc[max[0],:][j] > caminar[i][0].iloc[max[1],:][j]:
                orden_caminar.append(1)
            else:
                orden_caminar.append(0)
    return orden_correr,orden_caminar

# %%
def espejo(list_caminar,list_correr):
   for i in range(len(list_caminar)):
            #print(caminar_namesfile[i//4][0][columnas[i%4]][0])
            if list_caminar[i] == 0:
                caminar_namesfile[i//4][0][columnas[i%4]] = list(caminar_namesfile[i//4][0][columnas[i%4]][::-1])
                caminar_namesfile[i//4][0][columnas[i%4]] = list(caminar_namesfile[i//4][0][columnas[i%4]].reset_index(drop=True))
                #caminar_namesfile contiene la lista de todos los data set de caminar, es una lista de listas donde cada elemento tiene el dataframe y el nombre del archivo de donde se obtuvo
                #por eso el [i][0] esto representa un dataframe [columnas[i%4]] es una columna de ese dataframe y [::-1] invierte el orden de las filas
            if list_correr[i] == 0:
                correr_namesfile[i//4][0][columnas[i%4]] = list(correr_namesfile[i//4][0][columnas[i%4]][::-1])
                correr_namesfile[i//4][0][columnas[i%4]] = list(correr_namesfile[i//4][0][columnas[i%4]].reset_index(drop=True))



#%%
caminar_namesfile,correr_namesfile,max = redimensionar(caminar_namesfile,correr_namesfile,6)
orden_correr,orden_caminar = espejo_bool(correr_namesfile,caminar_namesfile)
espejo(orden_caminar,orden_correr)
#%%
for i in range(len(caminar_namesfile)):
    graficar(caminar_namesfile[i][0],caminar_namesfile[i][1])
    graficar(correr_namesfile[i][0],correr_namesfile[i][1])

# %%
#choose the 20 first rows of the dataframe

# %%
def plot_relations(archivo,titulo):
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(titulo, fontsize=20, fontweight='bold')
    fig.subplots_adjust(top=0.85)
    fig.subplots_adjust(hspace=0.8, wspace=0.5)

    #titles of the graph
    axs[0, 0].set_title('x_y_relation vs frames', fontsize=10,color='r')
    axs[0, 1].set_title(' area_relation vs frames', fontsize=10, color='r')
    axs[1, 0].set_title('x_y_relation vs area_relation',fontsize=10, color='r')

     #ejes x of the first subplot
    axs[0, 0].set_xlabel('Area_R')
    axs[0, 1].set_xlabel('Frames')
    axs[1, 0].set_xlabel('Frames')
    axs[0,0].set_ylabel('X_Y')
    axs[0,1].set_ylabel('Area_R')
    axs[1,0].set_ylabel('X_Y')
    
    #plot the data
    axs[0, 0].plot(range(len(archivo)), archivo['changecenter'], 'r')
    axs[0, 1].plot(range(len(archivo)), archivo['chancearea'], 'b')
    axs[1, 0].plot(archivo['changecenter'],archivo['chancearea'],'g')

 # %%
for i in range(len(caminar_namesfile_T)):
    plot_relations(caminar_namesfile_T[i][0],caminar_namesfile[i][1])
    plot_relations(correr_namesfile_T[i][0],correr_namesfile[i][1])



#share the link vidios alocated in drive with github 