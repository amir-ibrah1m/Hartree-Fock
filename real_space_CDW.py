import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import unitary_group

'''''
def Hartree(P,V,N):
   P_V = np.zeros([N,N],dtype=complex)
   for i in range(N):
      if i == 0:
        #print(P[i+1,i+1])
        P_V[i,i] = V*(P[i+1,i+1]+P[N-1,N-1])
      if i == N:
        P_V[i,i] = V*(P[i-1,i-1]+P[0,0])
      if i > 0:
         if i < (N-1):
            P_V[i,i] = V*(P[i-1,i-1]+P[i+1,i+1])
   return P_V
'''

def Hartree(P,V,N):
   P_V = np.zeros([N,N],dtype=complex)
   for i in range(N):
      if i == 0:
        P_V[i,i] = V*(P[i+1,i+1]+P[N-1,N-1])
      if i == (N-1):
        P_V[i,i] = V*(P[i-1,i-1]+P[0,0])
      if i > 0:
         if i < (N-1):
            P_V[i,i] = V*(P[i-1,i-1]+P[i+1,i+1])
   return P_V
   
   '''
def Fock(P,V,N):
    P_V = np.zeros([N,N],dtype=complex)
    for i in range(N):
      if i == 0:
        P_V[i,i+1] = V*P[i+1,i]
      if i == N:
        P_V[i,i-1] = V*(P[i-1,i])
      if i > 0:
         if i < (N-1):
            P_V[i,i+1] = V*P[i+1,i]
            P_V[i,i-1] = V*P[i-1,i]
    return P_V
    '''
def Fock(P,V,N):
    P_V = np.zeros([N,N],dtype=complex)
    for i in range(N):
      if i == 0:
        P_V[i,i+1] = V*P[i+1,i]
        P_V[0,N-1] = V*P_V[N-1,0]   #from pbcs?
      if i == (N-1):
        P_V[i,i-1] = V*(P[i-1,i])
        P_V[N-1,0] = V*P_V[0,N-1]   #from pbcs?
      if i > 0:
         if i < (N-1):
            P_V[i,i+1] = V*P[i+1,i]
            P_V[i,i-1] = V*P[i-1,i]
    #P_V[0,N-1] = V*P_V[N-1,0]
    #P_V[N-1,0] = V*P_V[0,N-1]
    return P_V


def single_particle(N, t):
    H_SP = np.zeros([N,N])
    for i in range(N):
      if i == 0:
        H_SP[i,i+1] = -t
      if i == (N-1):
        H_SP[i,i-1] = -t
      if i > 0:
         if i < (N-1):
            H_SP[i,i+1] = -t
            H_SP[i,i-1] = -t
    H_SP[0,N-1] = -t
    H_SP[N-1,0] = -t
    return H_SP


def energy(P, H_SP, H_H, H_F):
    
    KE = np.einsum('ij, ij ->', H_SP, P)
    E_H = np.einsum('ij, ij ->', P, H_H)/2
    E_F = - np.einsum('ij, ij ->', P, H_F)/2
    return KE + E_H + E_F


def aufbau(H, nu, save = False):
    
    P = np.zeros_like(H, dtype = complex)
    
    N = H.shape[0]
    n_elec = int(N*nu)
    
    eigs = np.zeros(N)
    evecs = np.zeros((N,N), dtype = complex)
    
                             
    eigs, evecs = np.linalg.eigh(H)


    #min_eig = eigs[0]
    min_index = 0
    filled_energy_indices = np.zeros(n_elec)
    
    #for l in range(n_elec):
     #   for i in range(N):
      #      eig = eigs[i]
       #     if eig < min_eig:
        #        min_eig = eig
         #       min_index = i
        #filled_energy_indices[l] = min_index
       # min_eig = math.inf
        #eigs[min_index]= min_eig
    
    for i in range(N):
        for j in range(N):
           #P[i,j] = gnd_state[i].conj()*gnd_state[j]
           
           for n in range(n_elec):
#              P[i,j] += (evecs[i,int(filled_energy_indices[n])])*np.conjugate(evecs[j,int(filled_energy_indices[n])])
               P[i,j] += (evecs[i,n])*np.conjugate(evecs[j,n])

    
    #print(np.trace(P))
    
    if save:
        return P, eigs, evecs
    else:
        return P



def random_pop_list(n_sym, n_elec, HFdim):
    """Randomly populate n_sym buckets with n_elec objects, with max HFdim in each bucket"""
    N=0
    iteration=0
    pop_list=np.zeros(n_sym,dtype=int)
    while N<n_elec:
      isym=np.random.randint(0,n_sym)
      if pop_list[isym]<HFdim:
        pop_list[isym]+=1
        N+=1
        iteration+=1
    return pop_list

def is_odd(num):
    return num & 0x1

#fix this
def input_P(N, nu, seed, symmetry_breaking):
    
    np.random.seed(seed)
    n_elec = N*nu
    pop_list = random_pop_list(N, n_elec, 1)
    P = np.zeros((N,N), dtype = complex)


    alpha = n_elec/N
    alpha = nu
    for i in range(N):
        P[i,i] = alpha

    beta_real = np.random.rand(1)[0]
    beta_imag = np.random.rand(1)[0]
    beta = beta_real + 1j*beta_imag
    for i in range(N):
      if i == 0:
        P[i,i+1] = beta
      if i == (N-1):
        P[i,i-1] = beta
      if i > 0:
         if i < (N-1):
            P[i,i+1] = beta
            P[i,i-1] = beta

    P[0,N-1] = beta
    P[N-1,0] = beta


    if symmetry_breaking:
       U = unitary_group.rvs(N)
       U_dag = np.linalg.inv(U)
       P = np.matmul(np.matmul(U_dag,P),U)
       return P
           
    else:
       return P
    

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def check_equal(a, b, rtol=1e-05, atol=1e-08):
    return np.allclose(a, b, rtol=rtol, atol=atol)

def diag_constant(a,tol=1e-3):
   diag_constant=True
   const = a[0][0]
   for i in range(len(a)):
      if (a[i][i] < (const-tol)) or (a[i][i] > (const+tol)):
         diag_constant=False
   return diag_constant

                                
      

   




t = 1
V = 100
N = 10
n_seeds = 10
nu = 1/2
ODA = False
n_iter = 100


H_SP = single_particle(N, t)

final_Ps = np.zeros((n_seeds,N,N), dtype=complex)

Es = np.zeros(n_seeds)
Es_old = np.zeros(n_seeds)
for seed in range(n_seeds):
   P = input_P(N, nu, seed, False)
   for iter in range(n_iter):
      H_H = Hartree(P, V, N)
      H_F  = Fock(P, V, N)
      E = energy(P, H_SP, H_H, H_F)
      #print(E)
      P_new = aufbau(H_SP + H_H - H_F, nu)
      if not ODA:
            P = P_new
            
      else:
            dP = P_new-P
            H_dP = Hartree(dP, Vq) - Fock(dP, Vq)
            
            a = np.sum(np.real(H_SP*dP)) + np.sum(np.real(H_dP*P))
            b = np.sum(np.real(H_dP*dP))/2
            
            if abs(b/(N1*N2)) < 1e-5:
                l = 1
            else:
                l = -a/(2*b)
                if l > 1 or l < 0:
                    l = 1
            P = P + l*dP
      if iter == (n_iter-2):
         Es_old[seed] = E
      if iter == (n_iter-1):
         Es[seed]=E
         final_Ps[seed] = P

final_CDW_Ps = np.zeros((n_seeds,N,N),dtype=complex)

Es_CDW = np.zeros(n_seeds)
Es_CDW_old = np.zeros(n_seeds)
for seed in range(n_seeds):
   P = input_P(N, nu, seed, True)
   for iter in range(n_iter):
      H_H = Hartree(P, V, N)
      H_F = Fock(P, V, N)
      E = energy(P, H_SP, H_H, H_F)
      #print(E)
      P_new = aufbau(H_SP + H_H - H_F, nu)
      if not ODA:
            P = P_new
            
      else:
            dP = P_new-P
            H_dP = Hartree(dP, Vq) - Fock(dP, Vq)
            
            a = np.sum(np.real(H_SP*dP)) + np.sum(np.real(H_dP*P))
            b = np.sum(np.real(H_dP*dP))/2
            
            if abs(b/(N1*N2)) < 1e-5:
                l = 1
            else:
                l = -a/(2*b)
                if l > 1 or l < 0:
                    l = 1
            P = P + l*dP
      if iter == (n_iter-2):
         Es_CDW_old[seed] = E
      if iter == (n_iter-1):
         Es_CDW[seed]=E
         final_CDW_Ps[seed]=P
    

print("---------")
print("Translationally symmetric")
print(Es)
print("Lowest energy symmetric solution")
print(min(Es))
if (abs(min(Es)-min(Es_old)) > 0.0005):
   print("Warning: Convergence not met")
else:
   print("Convergence met")
print("---------")
print("CDW")
print(Es_CDW)
print("Lowest energy CDW solution")
print(min(Es_CDW))
if (abs(min(Es_CDW)-min(Es_CDW_old)) > 0.0005):
   print("Warning: Convergence not met")
else:
   print("Convergence met")
print("---------")





final_P = final_Ps[min(range(len(Es)), key=Es.__getitem__)]
#print(final_P)
print("Final symmetric density matrix")
#print(final_P)
if check_symmetric(final_P) and diag_constant(final_P):
   print("IS symmetric")
else:
   print("NOT symmetric")
#print(final_Ps[min(range(len(Es)), key=Es.__getitem__)])

final_CDW_P = final_CDW_Ps[min(range(len(Es_CDW)), key=Es_CDW.__getitem__)]
print("Final CDW density matrix")
#print(final_CDW_P)
#print(final_CDW_Ps[min(range(len(Es_CDW)), key=Es_CDW.__getitem__)])
if check_symmetric(final_CDW_P) and diag_constant(final_CDW_P):
   print("IS symmetric")
else:
   print("NOT symmetric")

print("Symmetric P vs CDW P are equal:")
print(check_equal(final_P,final_CDW_P))


