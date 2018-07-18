import matplotlib.pyplot as plt
import numpy as np


def tabelle(datensatz, Name, Rundungen):  # i=Spalten j=Zeilen
    fobj_out = open(Name+".tex", "w")
    i_max, j_max = np.shape(datensatz)
    # fobj_out.write(r"\begin{table}"+"\n")
    # fobj_out.write(r"\centering"+"\n")
    # fobj_out.write(r"  \caption{}"+"\n")
    # fobj_out.write(r"  \label{}"+"\n")
    # fobj_out.write(r"\begin{tabular}{"+i_max*"S" +"} \n")
    # fobj_out.write(r"%"+i_max*"S" +"\n")
    # fobj_out.write(r"\toprule"+"\n \\\ \n")
    # fobj_out.write(r"\midrule"+"\n")
    #runden und in strings umwandeln
    for i in range(i_max):
        datensatz[i][:] = np.around(datensatz[i][:], decimals=Rundungen[i])
#    Werte=datensatz.astype(str)
#    print(Werte)

    for j in range(j_max):
        for i in range(i_max):
            formatierung='{0:.'+Rundungen[i].astype(str)+'f}'   # damit z.B. 0.30
            fobj_out.write(formatierung.format(datensatz[i,j])) # nicht 0.3 ist
            if(i<(i_max-1)):
                fobj_out.write("\t&\t")
        fobj_out.write(r"   \\ "+"\n")
    # [j,i] = np.shape(test)
    # for(i in )
    # fobj_out.write(r"\bottomrule"+"\n")
    # fobj_out.write(r"\end{tabular}"+"\n")
    # fobj_out.write(r"\end{table}"+"\n")
    fobj_out.close()

'''
test=np.array([[2.2960,3.0100,4.400,5.500,6.600,7.700,8.800,9.900],[1.1,2.2,3.3,4.4,5.5,6.6,7.7,8.8]])
test1=np.array([1,2,3,4,5])
print(np.shape(test))
print(test[0][7])
name="test"
print(np.around(test[0,:],decimals=3))
tabelle(test,name,np.array([2,1]))
Werte=test.astype(str)
print("&".join(test[0,:].astype(str)))

x=np.array(0.12345678)
print("seltsam"+'{0:.5f}'.format(x)+'ja')
print("round", np.around(x,5))

x=np.array(0.1200001)
print("seltsam"+'{0:.5f}'.format(x)+'ja')
print("round", np.around(x,5))
print("seltsam"+'{0:.5f}'.format(np.around(x,5))+'ja')
x="6"
mist='{0:.'+x+'f}'

print(mist.format(test[0,0]))
'''
