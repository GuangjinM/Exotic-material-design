
from __future__ import print_function
from fenics import *
import numpy as np,sys

#widget to get an interactive integral plot while auto to give out an interactive pop-up window plot
# %matplotlib auto
import matplotlib.pyplot as plt
import mshr
import cmath

sys.setrecursionlimit(20000)

from matplotlib.pyplot import figure
from matplotlib.ticker import FuncFormatter
from scipy.integrate import dblquad



#----------------------------------------------------------------
#Periodic boundary condition (CHANGED)
#----------------------------------------------------------------
class PeriodicBoundary(SubDomain):
    def __init__(self, vertices, tolerance=DOLFIN_EPS):
        """ vertices stores the coordinates of the 4 unit cell corners"""
        SubDomain.__init__(self, tolerance)
        self.tol = tolerance
        self.vv = vertices
        self.a1 = self.vv[1,:]-self.vv[0,:] # first vector generating periodicity
        self.a2 = self.vv[3,:]-self.vv[0,:] # second vector generating periodicity
        # check if UC vertices form indeed a parallelogram
        assert np.linalg.norm(self.vv[2, :]-self.vv[3, :] - self.a1) <= self.tol
        assert np.linalg.norm(self.vv[2, :]-self.vv[1, :] - self.a2) <= self.tol

    def inside(self, x, on_boundary):
        return bool((near(x[0], self.vv[0,0] + x[1]*self.a2[0]/self.vv[3,1], self.tol) or
                     near(x[1], self.vv[0,1] + x[0]*self.a1[1]/self.vv[1,0], self.tol)) and
                     (not ((near(x[0], self.vv[1,0], self.tol) and near(x[1], self.vv[1,1], self.tol)) or
                     (near(x[0], self.vv[3,0], self.tol) and near(x[1], self.vv[3,1], self.tol)))) and on_boundary)

    def map(self, x, y):
        if near(x[0], self.vv[2,0], self.tol) and near(x[1], self.vv[2,1], self.tol): # if on top-right corner
            y[0] = x[0] - (self.a1[0]+self.a2[0])
            y[1] = x[1] - (self.a1[1]+self.a2[1])
        elif near(x[0], self.vv[1,0] + x[1]*self.a2[0]/self.vv[2,1], self.tol): # if on right boundary
            y[0] = x[0] - self.a1[0]
            y[1] = x[1] - self.a1[1]
        else:   # should be on top boundary
            y[0] = x[0] - self.a2[0]
            y[1] = x[1] - self.a2[1]



#----------------------------------------------------------------
#constitutive law
#----------------------------------------------------------------

def eps(v):
    return sym(grad(v)) 

def sigma(v, Eps,rho,E,nu):
    lmbda = E*nu/((1+nu)*(1-nu))
    mu = E/(2*(1+nu))
    return rho*lmbda*tr(eps(v)+Eps)*Identity(2) + rho*2*mu*(eps(v)+Eps) #material is isotropic so we can use E and nu

#elasticity problem solving
def solve_pde(W,dx,Eps,rho,E,nu):
    v_,lamb_ = TestFunctions(W)
    dv, dlamb = TrialFunctions(W)
    w = Function(W)
    F = inner(sigma(dv,Eps,rho,E,nu), eps(v_))*dx
    a, L = lhs(F), rhs(F)
    a += dot(lamb_,dv)*dx + dot(dlamb,v_)*dx
    solve(a == L, w, [], solver_parameters={"linear_solver": "lu","preconditioner":"ilu"})   #cg, gmres(non symmetric),ilu,lu  "preconditioner":"ilu"
    (v, lamb) = split(w) 
    return v


#----------------------------------------------------------------
#returns the macroscopic strain for the 3 elementary load cases
#----------------------------------------------------------------
def macro_strain(j):
    Eps_Voigt = np.zeros((3,))
    Eps_Voigt[j] = 1
    return np.array([[Eps_Voigt[0], Eps_Voigt[2]/2.],
                    [Eps_Voigt[2]/2., Eps_Voigt[1]]])




#----------------------------------------------------------------
#Some math transformations   (CHANGED)
#----------------------------------------------------------------
#be calful od the use of np.array, it's not a ufl operator, we use as_vector/as_tensor instead
def stress2Voigt(s):
    return as_vector([s[0,0],s[1,1],s[0,1]])

def Voigt2stress(s):
    return as_tensor([[s[0],s[2]],[s[2],s[1]]])


def Voigt2tensor(c):  #definiton is wrong before
    return np.array([[[[c[0,0],c[0,2]],[c[0,2],c[0,1]]],[[c[2,0],c[2,2]],[c[2,2],c[2,1]]]],[[[c[2,0],c[2,2]],[c[2,2],c[2,1]]],[[c[1,0],c[1,2]],[c[1,2],c[1,1]]]]])


def tensor2UFL(c):
    return as_tensor([c[0,0,0,0],c[0,0,0,1],c[0,0,1,0],c[0,0,1,1],c[0,1,0,0],c[0,1,0,1],c[0,1,1,0],c[0,1,1,1],c[1,0,0,0],c[1,0,0,1],c[1,0,1,0],c[1,0,1,1],c[1,1,0,0],c[1,1,0,1],c[1,1,1,0],c[1,1,1,1]])


#Inverse of the tensor
def C_minus1(c):
    A=np.matrix(c)
    B=A.I
    return Voigt2tensor(B)



#----------------------------------------------------------------
#Subdomain definition related functions  (CHANGED)
#----------------------------------------------------------------
 


#Set values(label) to different subdomains
class K(UserExpression):
    def set_k_values(self, materials, k_0, k_1, **kwargs):
        self.materials = materials
        self.k_0 = k_0
        self.k_1 = k_1
        
    def eval_cell(self, values, x, cell):
        if self.materials[cell.index] == 0:
            values[0] = self.k_0
        else:
            values[0] = self.k_1




#----------------------------------------------------------------
#Function h(c)   (CHANGED)
#----------------------------------------------------------------

#second order tensor 
def varphi(i,j):
    varphi_test=np.zeros((2,2))
    varphi_test[i,j]=1       
    return np.array([[varphi_test[0,0],varphi_test[0,1]],[varphi_test[1,0],varphi_test[1,1]]]) 
    
def Prod_varphi(i,j,k,l):
    return as_tensor(np.tensordot(varphi(i,j),varphi(k,l),0))


def Type1_h(Chom,i,j,k,l):
    return inner(as_tensor(C_minus1(Chom)),Prod_varphi(i,j,k,l))

def Type2_h(Chom,i,j,k,l,m,n,p,q):
    test1=inner(as_tensor(C_minus1(Chom)),Prod_varphi(i,j,k,l))
    test2=inner(as_tensor(C_minus1(Chom)),Prod_varphi(m,n,p,q))
    return test1/test2 

def Type3_h(Chom,i,j,k,l):
    test1=inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(i,j,k,l))
    return test1

def Type4_h(Chom,i,j,k,l,m,n,p,q):
    test1=inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(i,j,k,l))
    test2=inner(as_tensor(Voigt2tensor(Chom)),Prod_varphi(m,n,p,q))
    return test1*test2


#----------------------------------------------------------------
#Update lsf (CHANGED)
#----------------------------------------------------------------

def update_lsf(g,V,th,kappa,lsf,deg):
    g_norm=np.sqrt(assemble(pow(g,2)*dx))
    g_project=Function(V)
    g_project.assign(project(g,V))
    lsf_new=Expression('1/N*(1/sin(th)*(sin((1-kappa)*th)*lsf+sin(kappa*th)*g/g_norm))',N=1,th=th,kappa=kappa,lsf=lsf,g=g_project,g_norm=g_norm,degree=deg)
    return lsf_new




#----------------------------------------------------------------
#definition of forth order tensor H   (CHANGED)               
#----------------------------------------------------------------

def comp_H(Gamma,E,nu):  #Gamma =gamma if material 0 and Gamma =1/gamma if material 1
    I2 = as_tensor([[1,0],[0,1]]) #second order identity tensor
    I4 = as_tensor([[[[1,0],[0,0]],[[0,1/2],[1/2,0]]],[[[0,1/2],[1/2,0]],[[0,0],[0,1]]]])
    alph=(1+nu)/(1-nu)
    bet=(3-nu)/(1+nu)
    return -1/E*((1-Gamma)/(1+alph*Gamma))*(4*I4-(1-Gamma*(alph-2*bet))/(1+bet*Gamma)*outer(I2,I2))



#----------------------------------------------------------------
#definition of D_TC    (CHANGED)         
#----------------------------------------------------------------
def comp_DTC(H,sig_mu0,sig_mu1,sig_mu2):
    D_TC0000=inner(H,outer(sig_mu0,sig_mu0))
    D_TC0001=D_TC0010=D_TC0100=D_TC1000=inner(H,outer(sig_mu0,sig_mu2))
    D_TC0011=D_TC1100=inner(H,outer(sig_mu0,sig_mu1))
    D_TC1010=D_TC0101=D_TC1001=D_TC0110=inner(H,outer(sig_mu2,sig_mu2))
    D_TC1110=D_TC1101=D_TC1011=D_TC0111=inner(H,outer(sig_mu1,sig_mu2))
    D_TC1111=inner(H,outer(sig_mu1,sig_mu1))
    return np.array([[[[D_TC0000,D_TC0001],[D_TC0010,D_TC0011]],[[D_TC0100,D_TC0101],[D_TC0110,D_TC0111]]],[[[D_TC1000,D_TC1001],[D_TC1010,D_TC1011]],[[D_TC1100,D_TC1101],[D_TC1110,D_TC1111]]]])



def Type1_DTJ(Chom,D_TC,i,j,k,l):  
    return -inner(as_tensor(np.tensordot(np.tensordot(C_minus1(Chom),D_TC,2),C_minus1(Chom),2)),as_tensor(Prod_varphi(i,j,k,l)))

def Type2_DTJ(Chom,D_TC,i,j,k,l,m,n,p,q):  
    test1=-inner(as_tensor(np.tensordot(np.tensordot(C_minus1(Chom),D_TC,2),C_minus1(Chom),2)),Prod_varphi(i,j,k,l))*inner(as_tensor(C_minus1(Chom)),Prod_varphi(m,n,p,q))
    test2=inner(as_tensor(np.tensordot(np.tensordot(C_minus1(Chom),D_TC,2),C_minus1(Chom),2)),Prod_varphi(m,n,p,q))*inner(as_tensor(C_minus1(Chom)),Prod_varphi(i,j,k,l))
    test3=inner(as_tensor(C_minus1(Chom)),Prod_varphi(m,n,p,q))*inner(as_tensor(C_minus1(Chom)),Prod_varphi(m,n,p,q))
    return (test1+test2)/test3

def Type3_DTJ(D_TC,i,j,k,l):
    test1=inner(as_tensor(D_TC),as_tensor(np.einsum('ij,kl->ijkl',varphi(i,j),varphi(k,l))))
    return test1

def Type4_DTJ(Chom,D_TC,i,j,k,l,m,n,p,q):
    test1=inner(as_tensor(D_TC),as_tensor(np.einsum('ij,kl->ijkl',varphi(i,j),varphi(k,l))))
    test2=inner(as_tensor(Voigt2tensor(Chom)),as_tensor(np.einsum('ij,kl->ijkl',varphi(m,n),varphi(p,q))))
    test3=inner(as_tensor(D_TC),as_tensor(np.einsum('ij,kl->ijkl',varphi(m,n),varphi(p,q))))
    test4=inner(as_tensor(Voigt2tensor(Chom)),as_tensor(np.einsum('ij,kl->ijkl',varphi(i,j),varphi(k,l))))
    return test1*test2+test3*test4


#----------------------------------------------------------------
#definition of th (CHANGED)
#----------------------------------------------------------------
def comp_th(threshold,g,lsf,dx):
    g_test=g+threshold
    lsf_test=lsf+threshold
    g_norm=np.sqrt(assemble(pow(g_test,2)*dx))
    lsf_norm=np.sqrt(assemble(pow(lsf_test,2)*dx))
    th=np.arccos(assemble(inner(g_test/g_norm,lsf_test/lsf_norm)*dx))
    return th




