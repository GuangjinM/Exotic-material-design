%************************************************************************************************
% DESCRIPTION
%************************************************************************************************

% Solve the 2D topology optimisation problem with the cost function from exotic material definition 
% Using FEniCS en Python 3.8


%************************************************************************************************
%CITATIONS AND REFERENCES
%************************************************************************************************

%ALGORITHEME:                      Sammuel AMSTUTZ (2010)
%HOMOGENISATION IMPLEMENTATION:    Jeremy Bleyer   (2016)
%LINE SEARCHE:                     Antoine Laurain (2017)


%************************************************************************************************
% HISTORY
%************************************************************************************************

% G.MOU  Code implementation                                                                 V1  11/2022
% G.MOU  Modified                                                                            V2  12/2022
% G.MOU  change the function definition of comp_DTC                                          V4  12/2022
% G.MOU  Re-arrange the code order / change the function definition of update_lsf()          V5  01/2023     
% G.MOU  Refinement debug on lsf_old & lsf_mat_old (update lsf_test=Function(V))             V6  01/2023
% G.MOU  Visualisation some functions/modified the definition of sig_mu1 and sig_mu2         V7  02/2023
% G.MOU  change all the pointwise definition to invoked function expressions/change the definition of Voigt2tensor/definition of identity forth order tensor     
                                                                                             V9  02/2023
%G.MOU   degree of project function space is set to 2 instead of 1. Threshold of level set value is introduced, the th() function is redefined.
         Topology of current iterations is saved in a xdmf file and will upload for the next iteration, the result of current optimization is saved in a txt file.
                                                                                             V13 04/2023
%G.MOU   repair the bug in line search                                                       V14 06/2023



from __future__ import print_function
from fenics import *
import numpy as np,sys

#widget to get an interactive integral plot while auto to give out an interactive pop-up window plot
#%matplotlib auto
import matplotlib.pyplot as plt
import mshr
from matplotlib.pyplot import figure
from matplotlib.ticker import FuncFormatter
from scipy.integrate import dblquad
# sys.setrecursionlimit(200000)

from Init import*
from definition import*


from matplotlib.ticker import MaxNLocator,MultipleLocator




#Initializations
my_type='Type4'  #four different types of function h
my_case='Cauchy elasticity'  #optimization problem


Nx,Ny,kappa0_initial,delta,delta2,lam,gamma,threshold,lsf=init(my_case)
kappa0=kappa0_initial
kappa=kappa0


#settings
resume=False    #
refinement=False
Linear=False  #elements type linear or quadric
stop=False
plotset=True 


#Mesh implementation
mesh=RectangleMesh(Point(0.0,0.0),Point(ax,ay),Nx,Ny,"crossed")
if Linear==True:
    deg=1
else: 
    deg=2

materials= MeshFunction('size_t', mesh,2)
dx = Measure("dx")(subdomain_data=materials,domain=mesh) 
vertices = np.array([[0, 0.],[ax, 0.],[ax, ay],[0, ay]])  #vertices used in periodic boundary condition

#Function spaces
Ve = VectorElement("CG", mesh.ufl_cell(), deg) #the number here represente for degree;
Re = VectorElement("R", mesh.ufl_cell(), 0) 
W = FunctionSpace(mesh, MixedElement([Ve, Re]), constrained_domain=PeriodicBoundary(vertices, tolerance=1e-10))  #vertices, tolerance=1e-10
V=FunctionSpace(mesh,'CG',deg)  #scalar valued linear lagrange elements   CG for 'continuous Galerkin'
Vvec = FunctionSpace(mesh, Ve) #vector valued lagrange elements      or     Vvec=VectorFunctionSpace(mesh,'CG',deg)
Vsig=TensorFunctionSpace(mesh,"DG",degree=deg,shape=(2,2))  #shape=(2,2) Vsig=TensorFunctionSpace(mesh,"DG",degree=0)
V4=TensorFunctionSpace(mesh,"DG",degree=deg,shape=(2,2,2,2))  #V4=TensorFunctionSpace(mesh,"DG",degree=0,shape=(2,2,2,2)) 
VolUnit=project(Expression('1.0',degree=2),V)


#Normalize the level set function
lsf.N=np.sqrt(assemble(pow(lsf,2)*dx))  
print(np.sqrt(assemble(pow(lsf,2)*dx)))
# #lsf=interpolate(lsf_init,V)
lsf_project=Function(V)
lsf_project.assign(project(lsf,V))
# print(lsf_project(0.5,0.5))
# print(lsf_project(0.75,0.5,))
# print(lsf_project(1,1))



#---------------------------------main loop --------------------------------------------------------
while It<ItMax and stop==False:

#for R0-orthotropic 
    # if It==2:
    #     kappa0_initial,delta,delta2=[0.1 ,0.8 ,0.98] #Line search criteria parameters; ls: line search iteration number; 
    #     kappa0=kappa0_initial
    #     kappa=kappa0

    # if It==29:
    #     kappa0_initial,delta,delta2=[0.05 ,0.8 ,0.98] #Line search criteria parameters; ls: line search iteration number; 
    #     kappa0=kappa0_initial
    #     kappa=kappa0
#for Cauchy elasticity 
    # if It==4:
    #     kappa0_initial,delta,delta2=[1 ,0.8 ,0.9] #Line search criteria parameters; ls: line search iteration number; 
    #     kappa0=kappa0_initial
    #     kappa=kappa0

    if refinement==False:
        
        if resume==True: 
            #upload the level set function for current iteration
            file = XDMFFile('previous_level_set_function.xdmf')
            lsf_project_loaded=Function(V)
            file.read_checkpoint(lsf_project_loaded,'lsf_project',0)
            lsf_project=lsf_project_loaded
    
        
        #Define subdomain: Omega(followed by the definition of level-set functions)
        class Omega_0(SubDomain):  #inclusion
            def inside(self, x, on_boundary):
                return lsf_project(x[0],x[1])<=threshold+tol

        materials.set_all(1)
        subdomain_0 = Omega_0()
        subdomain_0.mark(materials, 0)


        #rho assignment
        rho = K(materials,degree=0)
        rho.set_k_values(materials, 1, gamma)


        #elasticity problem solving
        Chom = np.zeros((3, 3))
        for (j, case) in enumerate(["Exx", "Eyy", "Exy"]):
            Eps = Constant(((0, 0), (0, 0))) 
            Eps.assign(Constant(macro_strain(j)))
            Sigma = np.zeros((3,))
            v=solve_pde(W,dx,Eps,rho,E,nu)     #return v dispalacement value another value lamb is not returned here since it will not be used later
        
            if j==0:   #for deformation loading of u11=epsilon_11 
                sig_mu0=sigma(v,Eps,rho,E,nu) 
            elif j==1: #for deformation loading of u22=epsilon_22
                sig_mu1=sigma(v,Eps,rho,E,nu)
            elif j==2: #for deformation loading of u12=epsilon_12
                sig_mu2=sigma(v,Eps,rho,E,nu)
            for k in range(3):
                Sigma[k] = assemble((stress2Voigt(sigma(v, Eps,rho,E,nu)))[k]*dx)/vol     #ufl format
            Chom[j, :] =Sigma
        print(np.array_str(Chom, precision=6))
        A=np.matrix(Chom)
        B=A.I
     

        #Get gamma* (denoted by Gamma here)
        Gamma = K(materials,degree=0)  #degree=0 or 1?
        Gamma.set_k_values(materials, gamma, 1/gamma)  #gamma, 1/gamma
  
   
        #4th order tensor H
        H=comp_H(Gamma,E,nu)                        
        
        #Topological derivative of C 
        D_TC=comp_DTC(H,sig_mu0,sig_mu1,sig_mu2)

        #cost function 
        volmat=assemble(VolUnit*dx(0))   

        h,D_TJ=init_type(my_type,my_case,Chom,D_TC)
    
        J[It]=h+lam*volmat/vol
        
        
        #D_TJ (question: how to implement tensor constraction of order 4)
        D_TJ=D_TJ+lam   



        #Function g
        g_parameter = K(materials,degree=0)
        g_parameter.set_k_values(materials, -1, 1)
        g = g_parameter*D_TJ
        g_norm=np.sqrt(assemble(pow(g,2)*dx)) #to normalize the g function, lsf is already normalized
        
        #Function th
        th=comp_th(threshold,g,lsf,dx)
        
      
        #----------------------------------Line search ---------------------------------------
        if It>0 and J[It]>J[It-1] and ls<ls_max :  # 'It' represent the current iteration
            ls+=1
            kappa*=delta
            lsf_project=lsf_old_project   #return the lsf function to the previous iteration

            lsf=update_lsf(g,V,th,kappa,lsf_project,deg)
            # lsf.N=np.sqrt(assemble(pow(lsf,2)*dx))
            # print('norm of lsf-------------------------------------------------')
            # print(np.sqrt(assemble(pow(lsf,2)*dx)))

            lsf_project=Function(V)
            lsf_project.assign(project(lsf,V))  

            file = XDMFFile('previous_level_set_function.xdmf')
            file.write_checkpoint(lsf_project,"lsf_project",0)
            file.close()


            print('**********line search iteration:%s'%ls)
            print('Function value:%.2f'%J[It])  
            print('Kappa:%s'%kappa)  
        
        else:
            if ls>0:
                print('**********out of line search')
                print('**********line search iteration:%s'%ls)
                print('Function value:%.2f'%J[It])  
                print('Kappa:%s'%kappa) 
                
            print('***********iteration number %s'%It)
            print('Function value:%.8f'%J[It])
            print('lambda * Volume fraction: %.8f'%(lam*volmat/vol))
            print('Kappa:%s'%kappa)
            print('Angle between g and lsf:%s'%(th*180/pi))
            print('Homogenised elasticity tensor:')
            print(np.array_str(Chom, precision=6))
            print(np.array_str(B, precision=6))

            def StoreFile(It):
                if It==0:
                    with open("/home/mou/OneDrive/Etude doctorale/FEniCS/Result data/Cauchy elasticity4/Test1_{}.txt".format(my_case),"w") as file:
                        file.write("")
                file=open("/home/mou/OneDrive/Etude doctorale/FEniCS/Result data/Cauchy elasticity4/Test1_{}.txt".format(my_case),"a")
                if ls>0:
                    file.write('\n\n**********out of line search \n')
                    file.write('**********line search iteration:%s\n'%ls)
                    file.write('Function value:%.2f\n'%J[It])
                    file.write('Kappa:%s\n'%kappa) 
                file.write('\n\n***********iteration number %s \n'%It)
                file.write('Function value:%.8f\n'%J[It])
                file.write('Volume fraction: %.8f\n'%(volmat/vol))
                file.write('Kappa:%s\n'%kappa)
                file.write('lambda:%s\n'%lam)
                file.write('Angle between g and lsf:%s\n'%(th*180/pi))
                file.write('Homogenised elasticity tensor:\n')
                file.write(np.array_str(Chom, precision=8))
                file.close()

            StoreFile(It)

            #store the plotting values
            J_plot.append(J[It])
            th_plot.append(th*180/pi)
            Vol_plot.append(volmat/vol)


            #Plot of subdomain

            print('Subdomain representation of current iteration:')
            plt.figure()
            plt.title('Iteration_{}'.format(It),fontsize=15,fontweight="bold",fontstyle='italic')
            plot(materials)
            plt.savefig('/home/mou/OneDrive/Etude doctorale/FEniCS/Result data/Cauchy elasticity4/Test1_{}.pdf'.format(It))
            plt.show()


            if plotset==True:
                print('lsf plot before update_lsf:')
                lsf_plot=Function(V)
                lsf_plot.assign(project(lsf,V))
                plt.figure()
                p=plot(lsf_plot)
                plt.colorbar(p)
                plt.show()

                print('g plot before update_lsf:')
                plt.figure()
                p=plot(g/g_norm)
                plt.colorbar(p)
                plt.show()
         
            
            #decrease or increase the line search step
            if ls==ls_max: kappa0=max(kappa*delta,0.1*kappa0_initial)
            if ls==0: kappa0=min(kappa/delta2,1)   #speed up the algorithm  

            #reset kappa and line search index
            ls,kappa,It,resume=[0,kappa0,It+1,True]    #after then, It-1 represent the current iteration number

            #store level set function
            lsf_old_project=lsf_project


            #update level set function
            lsf=update_lsf(g,V,th,kappa,lsf_project,deg)
            # lsf.N=np.sqrt(assemble(pow(lsf,2)*dx))
            # print('norm of lsf-------------------------------------------------------')
            # print(np.sqrt(assemble(pow(lsf,2)*dx)))
            
            lsf_project=Function(V)
            lsf_project.assign(project(lsf,V))  
          



            file = XDMFFile('previous_level_set_function.xdmf')
            file.write_checkpoint(lsf_project,"lsf_project",0)
            file.close()
            
            
            # stoppint criteria
            if (It>6 and max(abs(J[It-5:It-1]-J[It-1]))<abs(2.0*J[It-1]/(Nx**2))) or (It>6 and (max(abs(J[It-5:It-1]-J[It-1]))<abs(2.0*J[It-1]/(Nx**2))) and kappa==1):
                if th<=0.1:      #th<=5.72° if value is 0.1
                    stop=True
                    print(np.array_str(B, precision=6))   #print final result of C-1
                elif Nx>=300: 
                    refinement=False 
                    stop=True
                    print(np.array_str(B, precision=6))   #print final result of C-1
                else:
                    refinement=True
            # if It==4:
            #     stop=True


    else: #refinement=True
        print('********************************************Refinement******************************************************')
        Nx=Nx+40
        Ny=Ny+40

        mesh=RectangleMesh(Point(0.0,0.0),Point(ax,ay),Nx,Ny,"crossed")
    
        #Function spaces
        Ve = VectorElement("CG", mesh.ufl_cell(), deg) #the number here represente for degree;
        Re = VectorElement("R", mesh.ufl_cell(), 0) 
        W = FunctionSpace(mesh, MixedElement([Ve, Re]), constrained_domain=PeriodicBoundary(vertices, tolerance=1e-10))  #vertices, tolerance=1e-10
        V=FunctionSpace(mesh,'CG',deg)  #scalar valued linear lagrange elements   CG for 'continuous Galerkin'
        Vvec = FunctionSpace(mesh, Ve) #vector valued linear lagrange elements      or     Vvec=VectorFunctionSpace(mesh,'CG',1)
        Vsig=TensorFunctionSpace(mesh,"CG",degree=deg,shape=(2,2))  #shape=(2,2) Vsig=TensorFunctionSpace(mesh,"DG",degree=0)
        V4=TensorFunctionSpace(mesh,"CG",degree=deg,shape=(2,2,2,2))  #V4=TensorFunctionSpace(mesh,"DG",degree=0,shape=(2,2,2,2)) 
        VolUnit=project(Expression('1.0',degree=2),V)

        materials= MeshFunction('size_t', mesh,2)
        dx = Measure("dx")(subdomain_data=materials,domain=mesh)  

        refinement=False
        resume=False


#------------------------------------------------------------------------------
#PRINT RESULTS
#------------------------------------------------------------------------------
if plotset==True:
    plt.figure()
  
    plt.minorticks_on()
    plt.xlabel('Iteration number',fontsize=12)
    plt.ylabel('Cost function',fontsize=12)
    plt.title('Convergence history of cost function',fontsize=15,fontweight="bold",fontstyle='italic')
    plt.grid(b=True,linestyle='--', which='major') 
    plt.text(It-1,J_plot[It-1],'%1.4f'%(J_plot[It-1]),ha='left',va='top',fontweight="bold",fontsize=10)
    
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  #set xaxis as integer
    plt.gca().xaxis.set_major_locator(MultipleLocator(2))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
    
    plt.plot(J_plot,'ro-',color='r')
    plt.legend(loc='best')
    plt.show()



    plt.figure()
    plt.minorticks_on()
    plt.xlabel('Iteration number',fontsize=12)
    plt.ylabel('angle theta',fontsize=12)
    plt.title('Convergence history of angle theta',fontsize=15,fontweight="bold",fontstyle='italic')
    plt.grid(b=True,linestyle='--', which='major') 
    plt.text(It-1,th_plot[It-1],'%1.4f'%(th_plot[It-1]),ha='left',va='top',fontweight="bold",fontsize=10)
    
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  #set xaxis as integer
    plt.gca().xaxis.set_major_locator(MultipleLocator(2))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(1))

    plt.plot(th_plot,'ro-',color='r')
    plt.legend(loc='best')
    plt.show()



    plt.figure()
    plt.minorticks_on()            
    plt.ylabel('Volum constraint',fontsize=12)
    plt.title('Convergence history of volum constraint',fontsize=15,fontweight="bold",fontstyle='italic')
    plt.grid(b=True,linestyle='--', which='major') 
    plt.text(It-1,Vol_plot[It-1],'%1.4f'%(Vol_plot[It-1]),ha='left',va='top',fontweight="bold",fontsize=10)

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  #set xaxis as integer
    plt.gca().xaxis.set_major_locator(MultipleLocator(2))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
    
    plt.plot(Vol_plot,'ro-',color='r')
    plt.legend(loc='best')
    plt.show()
