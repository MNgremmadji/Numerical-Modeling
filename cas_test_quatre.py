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
D=1; dt=0.1## Pas de temps
T=1;t=np.arange(0,T,dt);m=len(t);sigma=5e-3
def nu(XX,YY):
    return D

## condition initiale
def rhozero(x,y):
    return (5/sigma)*np.exp(-((x-0.5)**2+(y-0.05)**2)/(2*sigma))

##### Definissons les fonctions maximum et minimum f+ et f-

def f_plus(x):
    return 0.5*(x+np.abs(x))
def f_moins(x):
    return 0.5*(-x+np.abs(x))

Ny=Nx=5
M=rectangle(Nx,Ny); N=M.nvol
X = M.centres[:, 0]
Y = M.centres[:, 1]
X, Y = np.meshgrid(X, Y)
## Les inconnues discrètes
rho=np.zeros((N,m)); c=np.zeros((N,m))
khi=[1];l=len(khi)
errl2=list()
h=list();eps=50;normtpsesp=0;nbr_iter=0 ## nombre d'iteration sur khi
while normtpsesp < eps:
    for yzx in range(l): 
        khi=khi1[yzx]
    ## Initialisation de rho
        for i in range(N):
            rho[i,0]=rhozero(M.centres[i][0],M.centres[i][1])
    ## c constant à chaque instant
        for j in range(m):
            for i in range(N):
                c[i,j]=M.centres[i][0] ## On prend la première coordonnée
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
### Definissons de la matrice diagonale contenant les mesures des volumes de contrôle
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

         # Norme L2 de la solution approché en espace et en temps
        w=np.zeros((m,1));w1=np.zeros((N,1))
        for j in range(m):
            for i in range(N):
                w1[i]=M.compute_vol()[i]*(rho[i,j])**2
            w[j]=np.sqrt(np.sum(w1))
        normtpsesp=np.linalg.norm(w,2) ## norme L2
        for j in range(m):
## la visualisation de la solution approchée rho à chaque temps comme une surface
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            surf=gca(projection='3d').plot_surface(X,Y,rho[:,j])
            plt.title('Solution numerique')
            plt.savefig("Surface_rho_casTest_cinq_{j}.png".format(j=j))
            plt.show()
## la visualisation de la solution approchée rho à chaque temps avec la fonction plotdiscrete
            fig = plt.figure()
            plotdiscrete(M,rho[:,j])
            plt.title('Solution numerique')
            plt.savefig("plotdiscrete_rho_casTest_cinq_{j}.png".format(j=j))
            plt.title('solution numerique')
            plt.show()
    nbr_iter=nbr_iter+1
