import numpy as np
import sys
from numpy import linalg as LA

def miller():
    h = int(input('enter h: '))
    k = int(input('enter k: '))
    l = int(input('enter l: '))
    print("miller_sum_squared",h**2+k**2+l**2)




def generate_miller(x, func , f1 ,f2):
    hkl = np.array([[0, 0, 0]])
    hkl_abs = np.array([0])
    for h in range(0,x+1):
        for k in range(0,x+1):
            for l in range(1,x+1):
                if ((h <= k) and (k <= l)):
                    hkl=np.append(hkl,[[h, k ,l]] , axis = 0 )
                    hkl_abs = np.append(hkl_abs, [h*h + k*k + l*l])
    sortarray = np.argsort(hkl_abs)
    hkl_sorted = hkl[sortarray,:]
    delete = np.array([0])
    for index, value in enumerate(hkl_sorted[:,1]):
            # print("abs=",func(f1 ,f2 , hkl_sorted[index,:]))
            if(func(f1 ,f2 , hkl_sorted[index,:]) == 0.0 ):
                delete = np.append(delete,index)

    # print("delete1 vorher:\n",delete)
    # print(hkl_sorted)
    hkl_sorted= np.delete(hkl_sorted, delete  , 0)
    # print("delete2 vorher:\n")
    # print(hkl_sorted)
    delete2 = np.array([0])
    for index, value in enumerate(hkl_sorted[:,1] ):
        if(index != 0):
            if(LA.norm(hkl_sorted[index,:]) == LA.norm(hkl_sorted[index - 1,:])):
                # print("eins=",LA.norm(hkl_sorted[index,:]))
                # print("zwei=",LA.norm(hkl_sorted[index - 1,:]))
                # print(index)
                delete2 = np.append(delete2,index)
    hkl_sorted = np.delete(hkl_sorted, delete2[1:]  , 0)
    # print("delete2 nachher:\n",delete2)
    # print(hkl_sorted)
    return hkl_sorted



def fluorit(f1, f2, arr):
    v1 = -1j * np.pi
    v2 = -1j * np.pi / 2
    h = arr[0]
    k = arr[1]
    l = arr[2]
    f1 = f1 * (1 + np.exp(v1 * (h + k)) + np.exp(v1 * (h + l)) + np.exp(v1 * (k + l)))
    # print( "h k l=", h, k, l)
    # print("f_1 =",f1)
    f2 = f2 * (np.exp(v2 *(3*h + 3 * k + l)) + np.exp(v2 * (3*h + k + 3 * l))
                + np.exp(v2 *(h + 3*k + 3*l))
                + np.exp(v2 *(h + k + 3 * l)) + np.exp(v2 *(h + 3 * k + l))
                + np.exp(v2 *(3 * h + k + l)))
    # print("f_2: ", f2)
    # print("abs=",round(abs(f2+f1),6))
    return(round(abs(f2+f1),6))

def s_s(f1, f2, arr):
    v1 = -1j * np.pi
    v2 = -1j * np.pi
    h = arr[0]
    k = arr[1]
    l = arr[2]
    f1 = f1 * (1 + np.exp(v1 * (h + k)) + np.exp(v1 * (h + l)) + np.exp(v1 * (k + l)))
    # print( "h k l=", h, k, l)
    # print("f_1 =",f1)
    f2 = f2 * (np.exp(v2 *(h +  k + l)) + np.exp(v2 * (2 * h + k + 2 * l))
                + np.exp(v2 *(h + 2 * k + 2 * l))
                + np.exp(v2 *(2 * h + 2 * k +   l)))
# print("f_2: ", f2)
    # print("abs=",round(abs(f2+f1),6))
    return(round(abs(f2+f1),6))

print(generate_miller(8,s_s,1,2))
# for index, value in enumerate(hkl[:,1]):
#       print(fluorit(1.0, 2.0, hkl[index,:]))
#       print("nächstes: hkl: \n\n")

# print("array mit alles: ", generate_miller(3, fluorit ,1 , 2), "\n\n\n")
# while(True):
#     miller()
