# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 21:34:33 2018

@author: Auxence
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 19:12:39 2018

@author: Auxence
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 08:10:58 2018

@author: Auxence
"""
import numpy as np
import matplotlib.pylab as plt
from edgestruct import *
import scipy.sparse as spsp
import scipy.sparse.linalg as spla
from matplotlib.ticker import LinearLocator, FormatStrFormatter


## Données du problème
D=1;khi=0.0; dt=0.05## Pas de temps
T=0.1;t=np.arange(0,T,dt);m=len(t)
def nu(XX,YY):
    return D

## condition initiale
def rhozero(x,y):
    return np.cos(np.pi*x)*np.cos(np.pi*y)
def c_zero(x,y):
    return (1/(1+2*np.pi**2))*np.cos(np.pi*x)*np.cos(np.pi*y)

## Definissons les fonctions maximum et minimum f+ et f-

def f_plus(x):
    return 0.5*(x+np.abs(x))
def f_moins(x):
    return 0.5*(-x+np.abs(x))
## ## La solution exacte est donnée: rho(t,x,y)=np.exp(-2*np.pi**2*t)*np.cos(np.pi*x)*np.cos(np.pi*y)
def sol_exacte_rho(t,x,y):
    return np.exp(-2*np.pi**2*t)*np.cos(np.pi*x)*np.cos(np.pi*y)
## La solution exacte c
def c_exacte(t,x,y):
    return (1/(1+2*np.pi**2))*np.exp(-2*np.pi**2*t)*np.cos(np.pi*x)*np.cos(np.pi*y)

Nn=[5];
l=len(Nn)
errl2=list()
h=list()
for k in range(l):
    Nx=Nn[k]
    Ny=Nx
    M=rectangle(Nx,Ny); N=M.nvol
    X = M.centres[:, 0]
    Y = M.centres[:, 1]
    X, Y = np.meshgrid(X, Y)
   ## Les inconnues discrètes
    rho=np.zeros((N,m)); c=np.zeros((N,m)) 
    
    ## Initialisation de rho et c
    for i in range(N):
        rho[i,0]=rhozero(M.centres[i][0],M.centres[i][1])
        c[i,0]=c_zero(M.centres[i][0],M.centres[i][1])
    for j in range (1,m): ## itération en temps
        
## Definissons la matrice du terme de diffusion
        
    ## La contribution des arêtes intérieur
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
    # La contributu=ion des arêtes neumann extérieures
        kb = M.Kbnd.tolist()
        I_A+=kb
        J_A+=kb
        D_A+=(M.mes_b*0.0).tolist() ## condition de neumann homogène
    ## Assemblage de la matrice
        Adiff=spsp.csr_matrix((D_A, (I_A, J_A)))
    ## Definissons de la matrice diagonale contenant les mesures des volumes de contrôle
        Id=np.diag((M.compute_vol()))
    ### Definissons la Matrice de convection
        ki = M.Kin.tolist()
        li = M.Lin.tolist()
        k1=0.0*M.dKL;k2=0.0*M.dKL
        for i in range(len(k1)):
            k1[i]=f_plus(c[M.Lin[i],j-1]-c[M.Kin[i],j-1])
        for i in range(len(k1)):
            k2[i]=f_moins(c[M.Lin[i],j-1]-c[M.Kin[i],j-1])
        ccc1=(khi*M.mes_i*k1/M.dKL).tolist()
        ccc2=(-khi*M.mes_i*k2/M.dKL).tolist()
        ccc4=(khi*M.mes_i*0.5*k1/M.dKL).tolist()
        ccc3=(-khi*M.mes_i*k2/M.dKL).tolist()
        D_A1=ccc1+ccc2+ccc3+ccc4
        I_A1 = ki+ki+li+li
        J_A1= ki+li+ki+li
    ## La prise en compte des arêtes extérieures de type neumanne homogène
        kb = M.Kbnd.tolist()
        I_A1+=kb
        J_A1+=kb
        D_A1+=(M.mes_b*0.0).tolist() ## Condition de neumann homogène
    ## Assemblage de la matrice de convection
        Aconv=spsp.csr_matrix((D_A1, (I_A1, J_A1)))
        A=Id+dt*(Adiff+Aconv) 
        bib=np.zeros((N)) ## Le second membre de la première équation
        for i in range(N):
            bib[i]=M.compute_vol()[i]*rho[i,j-1]
        s=np.linalg.solve(A,bib) ## La solution pour le premier schéma (en rho)
        for i in range(N):    
            rho[i,j]=s[i]
        B=Id+Adiff   
        bib1=np.zeros((N)) ## Le second membre de la deuxième équation 
        for i in range(N):
            bib1[i]=M.compute_vol()[i]*rho[i,j]
        s2=np.linalg.solve(A,bib) ## La solution pour le second schéma (en rho)
        for i in range(N):    
            c[i,j]=s2[i]
## Solution exacte de rho
    solexac=np.zeros((N,m))
    erreur=np.zeros((N,m))
    for j in range(m):
        for i in range(N):
            solexac[i,j]=sol_exacte_rho(t[j],M.centres[i][0],M.centres[i][1])
    erreur=solexac-rho
         # Norme L2 de l'erreur en espace
    w=np.zeros((N,1))
    for i in range(N):
        w[i]=M.compute_vol()[i]*(erreur[i,j])**2
    Err_l2=np.sqrt(np.sum(w)) ## La norme L2 de l'erreur pour chaque Nx
    errl2.append(Err_l2) ## liste contenant les normes L2 de l'erreur en espace
    h1=1/Nx ## pas h de l'espace pour Nx donné
    h.append(h1) ## on collectionne tout les pas pour chaque Nx dans une liste
errl2=np.asarray(errl2) ## On transforme la liste en un array
h=np.asarray(h)
    
    ##norme L2 en temps L2 en espace
normtpsesp=np.linalg.norm(errl2,2) ## norme L2 en temps
         
for j in range(m):
## Aperçu de l'erreur à chaque instant avec plotdiscret
    fig = plt.figure()
    plotdiscrete(M,erreur[:,j])
    plt.title('erreur commise')
    plt.savefig("plotdiscrete_Erreur_casTest_deux{j}.png".format(j=j))
    plt.show()
## Aperçu de l'erreur à chaque instant comme surface
    fig=plt.figure()
    surf=gca(projection='3d').plot_surface(X,Y,erreur[:,j])
    plt.title('erreur commise')
    plt.savefig("Surface_Erreur_casTest_deux_{j}.png".format(j=j))
    plt.show()
     
## la visualisation de la solution approchée rho comme une surface
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf=gca(projection='3d').plot_surface(X,Y,rho[:,j])
    plt.title('Solution numerique')
    plt.savefig("Surface_rho_casTest_deux_{j}.png".format(j=j))
    plt.show()
    ## solution numérique pour c
    fig = plt.figure()
    ax = fig.gca(projection='3d')
        # Plot the surface.
    surf=gca(projection='3d').plot_surface(X,Y,c[:,j])
    plt.title('Solution numerique')
    plt.savefig("Surface_c_casTest_deux_{j}.png".format(j=j))
    plt.show()
## la visualisation de la solution approchée rho avec la fonction plotdiscrete
    fig = plt.figure()
    plotdiscrete(M,rho[:,j])
    plt.title('Solution numerique')
    plt.savefig("plotdiscrete_rho_casTest_deux_{j}.png".format(j=j))
    plt.title('solution numerique')
    plt.show()
## la visualisation de la solution approchée c avec la fonction plotdiscrete
    fig = plt.figure()
    plotdiscrete(M,c[:,j])
    plt.title('Solution numerique')
    plt.savefig("plotdiscrete_c_casTest_deux_{j}.png".format(j=j))
    plt.title('solution numerique')            
