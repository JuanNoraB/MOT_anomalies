import pandas
import os
import math

#print csv files in the current directory

#read a csv
#df = pandas.read_csv("casa_k.csv")
#just take the 177 rowe of df
#df = df.iloc[:177]
#df = pandas.read_csv("/home/juanchx/Documentos/'prueba deepsort'/casa_k.csv")
#create a nuevo dataframe with thsame columns

#across for the df
#f

def two_variables(df):
    indx=0
    df1 = pandas.DataFrame(columns=["changecenter","chancearea","target"])
    for i in df.index[:-1]:
        centeri = (df.iloc[i][0] + (df.iloc[i][2]/2), df.iloc[i][1] + (df.iloc[i][3]/2))
        centerip1 = (df.iloc[i+1][0] + (df.iloc[i+1][2]/2), df.iloc[i+1][1] + (df.iloc[i+1][3]/2))
        Areai = df.iloc[i][2]*df.iloc[i][3]
        Areaip1 = df.iloc[i+1][2]*df.iloc[i+1][3]
        change_center= ( ( (max(centeri[0],centerip1[0]))/(min(centeri[0],centerip1[0]))  )  + ( (max(centeri[1],centerip1[1]))/(min(centeri[1],centerip1[1])) )  )/2 
        changa_Area = max(Areai,Areaip1)/min(Areai,Areaip1) 
        #write the rows of the new dataframe df1
        df1.loc[indx]=[change_center,changa_Area,0]
        indx+=1  
        #print(change_center,changa_Area)
        #print(df.iloc[i][0])
        #print(df.loc[i])
        #print(df.loc[i]["x1"])
        #print(df.loc[i]["y1"])
        #number of rows of df1
    #print(df1.shape[0])
    #print(df.shape[0])
    return df1


#list fo all file .csv in the current directory
files = [f for f in os.listdir('.') if f.endswith('.csv')]
for i in files:
    #read a csv
    df = pandas.read_csv(i)
    #just take the 178 rowe of df
    df = df.iloc[:177]
    #create the new dataframe
    df1= two_variables(df)
    #save the new dataframe
    name =i
    name = name[ :name.rfind('.')]
    df1.to_csv(f"F_{i}",index=False)