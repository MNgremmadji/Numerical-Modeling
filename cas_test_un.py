# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 11:59:04 2018

@author: Auxence
"""
import numpy as np
import matplotlib.pylab as plt
from edgestruct import *
from mpl_toolkits.mplot3d import Axes3D
import scipy.sparse as spsp
import scipy.sparse.linalg as spla
from matplotlib.ticker import LinearLocator, FormatStrFormatter


Nn=[11]
l=len(Nn)
errl2=list()
h=list()
for i in range(l):
    Nx=Nn[i]
    Ny=Nx## Nx,Ny: Number of points discretized on absciss and on ordone
    M=rectangle(Nx,Ny);## We create a rectangle with Nx X Ny dimension
    N=M.nvol ## Number of the volum of control
    D=1 ## Coefficient of the diffusion
    #X=np.linspace(0,1,N+2);Y=np.linspace(0,1,N+2); XX,YY=np.meshgrid(X,Y) ## Discretization of space
    ## Coeffiient of diffusion 
    def nu(XX,YY):
        return D
    ## initial data
    def rhozero(XX,YY):
        return np.cos(np.pi*XX)*np.cos(np.pi*YY)
    def czero(XX,YY):
        return (1/(1+2*np.pi**2))*np.cos(np.pi*XX)*np.cos(np.pi*YY)
    ## implementaion of scheme
    
    ## We compute matrix from diffusion terme
    ## Contribution of interne edge
    ki = M.Kin.tolist() 
    li = M.Lin.tolist()
    nu_i=0.0*M.dKL
    for i in range(len(nu_i)):
        nu_i[i]=nu(M.xs_i[i][0],M.xs_i[i][1])
    pos = (M.mes_i*nu_i/M.dKL).tolist()
    neg = (-M.mes_i*nu_i/M.dKL).tolist()
    D_A = pos+neg+neg+pos
    I_A = ki+ki+li+li
    J_A = ki+li+ki+li
    ## Contribution of bound edge
    kb = M.Kbnd.tolist()
    I_A+=kb
    J_A+=kb
    D_A+=(M.mes_b*0.0).tolist()
    ## We assemble the matrix of diffusion with sparse method
    Adiff=spsp.csr_matrix((D_A, (I_A, J_A)))
## We compute diagonal matrix contains the volum of each K
    Id=np.diag((M.compute_vol()))
    A=Id+Adiff
## We compute the second member b
    b=np.zeros((N))
    for i in range(N):
        b[i]=M.compute_vol()[i]*rhozero(M.centres[i][0],M.centres[i][1])
        ## We compute numeric solution
    num_sol=np.linalg.solve(A,b)
    ## visualisation solution numérique avec la fonction plotdiscrete
    fig = plt.figure()
    plotdiscrete(M, num_sol)
    plt.title('solution numerique')
    plt.savefig("aaa1.png".format(i=i))
    ## visualisation de la solution numérique comme une surface *
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = M.centres[:, 0]
    Y = M.centres[:, 1]
    X, Y = np.meshgrid(X, Y)
        # Plot the surface.
    surf=gca(projection='3d').plot_surface(X,Y,num_sol)
    plt.title('solution numerique')
    plt.show()
    plt.savefig("bbb1.png".format(i=i))
    ## Exacte solution 
    exact_sol=np.zeros((N))
    for i in range(N):
        exact_sol[i]=czero(M.centres[i][0],M.centres[i][1])
        ## Error of the scheme err
    err=exact_sol-num_sol
    ## Visualisation de l'erreur avec la fonction plot discrète
    fig=plt.figure()
    plotdiscrete(M,err)
    plt.title('erreur commise')
    plt.savefig("ccc1.png".format(i=i))
#    ## visualisation de l'erreur definie par une surface
    fig=plt.figure()
    surf=gca(projection='3d').plot_surface(X,Y,err)
    plt.title('erreur commise')
    plt.show()
    plt.savefig("ddd1".format(i=i))
    # Norme L2 de l'erreur
    w=np.zeros((N,1))
    for i in range(N):
        w[i]=M.compute_vol()[i]*(err[i])**2
    Err_l2=np.sqrt(np.sum(w)) ## La norme L2 de l'erreur pour chaque Nx
    errl2.append(Err_l2) ## liste contenant les normes L2 de l'erreur
    h1=1/Nx ## pas h pour Nx donné
    h.append(h1) ## on collectionne tout les pas dans une liste
#╚errl2=np.asarray(errl2) ## On transforme la liste en un array
#h=np.asarray(h) ## On transforme la liste en array
#fig=plt.figure
#plot(np.log(1./np.sqrt(Nn)),np.log(errl2),'r') 
#plt.title('La norme L2 de l erreur')
#plt.savefig("normeL2erreur")