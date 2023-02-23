
from curses.ascii import CR
import numpy as np
import scipy.linalg as linAlg
import matplotlib.pyplot as plt

# here defines the silicon lattice ( fcc version, 8 atoms in unit cell, space group 227 )
# atoms position:
#   [0,0,0] [1/2,1/2,0] [0,1/2,1/2] [1/2,0,1/2]  No. 0,1,2,3
#   [1/4,1/4,1/4] [3/4,3/4,1/4] [3/4,1/4,3/4] [1/4,3/4,3/4]   No.4,5,6,7

class hamilNaive_test:
    # E = H - mu
    def __init__(self, mu=1, t1 = 1):
        self.mu, self.t1 = mu, t1
        self.H0 = np.matrix([-self.mu])
        self.Hx = np.matrix([self.t1])

    # Hx[]
    def toGreensFunc(self,Nx,Ny):
        # to z direction
        Ox, Oy = np.matrix(np.eye(Nx)), np.matrix(np.eye(Ny)) #on site matrix
        Tx, Ty = np.matrix(np.diag([1]*(Nx-1),1)), np.matrix(np.diag([1]*(Ny-1),1)) # hopping nearest neighbour
        
        hXsmall = np.kron(Ox,self.H0) + np.kron(Tx,self.Hx) + np.kron(Tx,self.Hx).H     # 000, 100, -100
        temp = np.kron(Ox,self.Hx)       # 010, 110, -1-10
        H0Iter = np.kron(Oy,hXsmall) + np.kron(Ty,temp) + np.kron(Ty,temp).H             # layer hamiltonian

        H1Iter = np.kron(Oy,np.kron(Ox,self.Hx)) # + np.kron(Ty,np.kron(Tx,self.Hxyz)) + np.kron(Ty,np.kron(Tx,self.Hxyz)).H


        #generate coupling matrix
        #coupling index
        leftCoup = np.array([1])
        rightCoup = np.array([1])
        tLeft = np.kron(np.eye(Nx*Ny),leftCoup)                         #lead2Scatter
        tRight = np.kron(np.eye(Nx*Ny),rightCoup).transpose()           #Scatter2Lead

        return H0Iter, H1Iter, tLeft, tRight

class hamilCubicNaive:
    # E = H - mu
    def __init__(self, mu=1, t1 = 1, t0=1):
        self.mu, self.t1, self.t0 = mu, t1, t0
        temp = np.matrix(np.zeros((8,8)))
        temp[0,1]=temp[0,2]=temp[0,3]=temp[1,2]=temp[1,3]=temp[2,3]=temp[6,7]=temp[5,6]=temp[5,7]=temp[4,5]=temp[4,6]=temp[4,7] = 1
        temp2 = np.matrix(np.zeros((8,8)))
        temp2[0,4]=temp2[1,4]=temp2[1,5]=temp2[2,4]=temp2[2,7]=temp2[3,4]=temp2[3,6] = 1
        self.H0 = np.diag([1]*8)*(-self.mu) + temp*self.t1 + (temp*self.t1).H + temp2*self.t0 + (temp2*self.t0).H
        self.Hx  = np.zeros((8,8))
        self.Hx[0,1]=self.Hx[0,3]=self.Hx[2,3]=self.Hx[4,5]=self.Hx[4,6]=self.Hx[2,1]=self.Hx[7,5]=self.Hx[7,6] = self.t1
        self.Hx[2,5]=self.Hx[2,6] = self.t0
        self.Hy = np.zeros((8,8))
        self.Hy[0,1]=self.Hy[0,2]=self.Hy[4,5]=self.Hy[4,7]=self.Hy[3,1]=self.Hy[3,2]=self.Hy[6,5]=self.Hy[6,7] = self.t1
        self.Hy[3,5]=self.Hy[3,7] = self.t0
        self.Hz = np.zeros((8,8))
        self.Hz[0,2]=self.Hz[0,3]=self.Hz[1,2]=self.Hz[1,3]=self.Hz[4,6]=self.Hz[4,7]=self.Hz[5,6]=self.Hz[5,7] = self.t1
        self.Hz[1,6]=self.Hz[1,7] = self.t0
        self.Hxy = np.zeros((8,8))
        self.Hxy[0,1]=self.Hxy[4,5] = self.t1
        self.Hxy[0,5] = self.t0
        self.Hxpy = np.zeros((8,8))
        self.Hxpy[2,3]=self.Hxpy[7,6] = self.t1
        self.Hyz = np.zeros((8,8))
        self.Hyz[0,2]=self.Hyz[4,7] = self.t1
        self.Hyz[0,7] = self.t0
        self.Hypz = np.zeros((8,8))
        self.Hypz[3,1]=self.Hypz[6,5] = self.t1
        self.Hzx = np.zeros((8,8))
        self.Hzx[0,3]=self.Hzx[4,6] = self.t1
        self.Hzx[0,6] = self.t0
        self.Hzpx = np.zeros((8,8))
        self.Hzpx[1,2]=self.Hzpx[5,7] = self.t1


    # Hx[]
    def toGreensFunc(self,Nx,Ny):
        # to z direction
        Ox, Oy = np.matrix(np.eye(Nx)), np.matrix(np.eye(Ny)) #on site matrix
        Tx, Ty = np.matrix(np.diag([1]*(Nx-1),1)), np.matrix(np.diag([1]*(Ny-1),1)) # hopping nearest neighbour
        
        hXsmall = np.kron(Ox,self.H0) + np.kron(Tx,self.Hx) + np.kron(Tx,self.Hx).H     # 000, 100, -100
        temp = np.kron(Ox,self.Hy) + np.kron(Tx,self.Hxy) + np.kron(Tx,self.Hxpy).H      # (+y) 010, 110, 1-10
        H0Iter = np.kron(Oy,hXsmall) + np.kron(Ty,temp) + np.kron(Ty,temp).H             # layer hamiltonian

        H1Iter = np.kron(Oy,np.kron(Ox,self.Hz)) # (+z)
        H1Iter +=  np.kron(Oy,np.kron(Tx,self.Hzx)) + np.kron(Oy,np.kron(Tx.H,self.Hzpx))
        H1Iter += np.kron(Ty,np.kron(Ox,self.Hyz)) + np.kron(Ty,np.kron(Ox,self.Hypz)).H

        #generate coupling matrix
        #coupling index
#         leftCoup = np.array([1,1,1,1,1,1,1,1])
#         rightCoup = np.array([1,1,1,1,1,1,1,1])
        leftCoup = np.array([1,1,1,1,0,0,0,0])
        rightCoup = np.array([0,0,0,0,1,1,1,1])
        tLeft = np.kron(np.eye(Nx*Ny),leftCoup)                         #lead2Scatter
        tRight = np.kron(np.eye(Nx*Ny),rightCoup).transpose()           #Scatter2Lead

        return H0Iter, H1Iter, tLeft, tRight

    def toHk(self,k):
        Hk = np.matrix(np.zeros((8,8)),dtype='complex128')
        Hk += self.H0 * 0.5
        Hk += self.Hx*np.exp(1j*2*np.pi*(k@np.array([1,0,0]))) 
        Hk += self.Hy*np.exp(1j*2*np.pi*(k@np.array([0,1,0])))
        Hk += self.Hz*np.exp(1j*2*np.pi*(k@np.array([0,0,1]))) 
        Hk += self.Hxy*np.exp(1j*2*np.pi*(k@np.array([1,1,0])))
        Hk += self.Hyz*np.exp(1j*2*np.pi*(k@np.array([0,1,1]))) 
        Hk += self.Hzx*np.exp(1j*2*np.pi*(k@np.array([1,0,1])))
        Hk += self.Hxpy*np.exp(1j*2*np.pi*(k@np.array([1,-1,0])))
        Hk += self.Hypz*np.exp(1j*2*np.pi*(k@np.array([0,1,-1]))) 
        Hk += self.Hzpx*np.exp(1j*2*np.pi*(k@np.array([-1,0,1])))   

        Hk += Hk.H
        return Hk

    def bdStructrue(self, n=30):
        #[0,0,0]->[1,0,0]->[1,1,0]->[0,0,0]->[1,1,1]->[1,1,0]
        # gamma, x, m, gamma, R, m 
        # X1:starting pt, X2:ending pt, n: number of pts
        def lineDisperse(X1,X2,xPt,n,ax):
            Klist,Elist = [],[]
            dk = np.linalg.norm(X2-X1)/n
            for i in range(n):
                K = (X2-X1)/n*i + X1
                H = self.toHk(K)
                e,_ = np.linalg.eigh(H)
                Elist.append(e)
                Klist.append(dk*i+xPt)
            ax.plot(Klist,Elist,'bo')
            ax.axvline(x=Klist[-1],c='k')
            return Klist[-1]

        G,X,M,R = np.array([0,0,0]),np.array([1/2,0,0]),np.array([1/2,1/2,0]),np.array([1/2,1/2,1/2])
        
        fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(16,8))
        kend=0
        kend= lineDisperse(G,X,kend,n,ax)
        kend= lineDisperse(X,M,kend,n,ax)
        kend= lineDisperse(M,G,kend,n,ax)
        kend= lineDisperse(G,R,kend,n,ax)
        kend= lineDisperse(R,M,kend,n,ax)

def normalLeadSelfEnergy(e,Nx,Ny,eta):
    mu, t = 0, -1
    E = (e+eta*1j)*np.eye(Nx)
    H0 = np.matrix(-mu*np.eye(Nx)) + np.matrix(np.diag([1]*(Nx-1),t)) + np.matrix(np.diag([1]*(Nx-1),t)).H
    H0 = np.matrix(np.kron(np.eye(Ny),H0)) + t*np.matrix(np.kron(np.diag([1]*(Ny-1),1), np.eye(Nx))) + t*np.matrix(np.kron(np.diag([1]*(Ny-1),1), np.eye(Nx))).H
    H1 = np.matrix(t*np.eye(Nx*Ny))

    H1inv = np.linalg.inv(H1)
    E = np.kron(np.eye(Ny),E)
    temp1 = np.concatenate([H1inv@(E-H0), -H1inv@H1.H],axis=1)
    temp2 = np.concatenate([np.eye(Nx*Ny),np.zeros((Nx*Ny,Nx*Ny))],axis=1)
    TT = np.concatenate([temp1,temp2],axis=0)
    TT = TT@TT

    eigVal, eigVec = linAlg.eig(TT) # val[i] * vec[:,i] = TT * vec[:,i]
    order = np.argsort(np.abs(eigVal))
    v2 = eigVec[:,order]
    S1, S2 = v2[0:Nx*Ny,0:Nx*Ny], v2[Nx*Ny:2*Nx*Ny,0:Nx*Ny]
#     S1, S2 = v2[0:Nx*Ny,0:Nx*Ny], v2[0:Nx*Ny,Nx*Ny:2*Nx*Ny]

    gr = np.linalg.inv(E - H0 - H1*S1*np.linalg.inv(S2))

    return np.matrix(gr)


def dIdVE(e, Nz, Nx,Ny, sys, eta=0.0000001,coupling=0.1,wdLead=False):
    # generate G00,G01,G10,G11 according to H0, H1, at energy E
    c = coupling  

    H0,H1,cLeft,cRight = sys.toGreensFunc(Nx,Ny)
    N = H0.shape[0]
    
    if wdLead:
        SigmaL = normalLeadSelfEnergy(0,Nx,Ny,eta)
    else:
        SigmaL = normalLeadSelfEnergy(e,Nx,Ny,eta)
    SigmaR = SigmaL[::-1,::-1]    #in case M-field
    #     print(np.imag(SigmaL), np.imag(SigmaR))

    SigmaL, SigmaR = c**2*cLeft.transpose()@SigmaL@cLeft, c**2*cRight@SigmaR@cRight.transpose()
    
    E = (e+eta*1j)*np.eye(N)
#     grii = np.linalg.inv(E-H0)
    grii = np.linalg.inv(E-H0-SigmaL )
    gr1i = gri1 = gr11 = np.copy(grii)

    GammaL, GammaR = 1j*(SigmaL-SigmaL.H), 1j*(SigmaR-SigmaR.H)
#     Sigma_total = np.kron([[1,0],[0,0]],SigmaL) + np.kron([[0,0],[0,1]],SigmaR)
    Sigma_total = np.kron([[0,0],[0,1]],SigmaR)
    for i in range(Nz):
        grii = np.linalg.inv(E - H0 - H1@grii@H1.H)
        gr1i = gr1i @ H1.H @ grii
        gr11 = gr11 + gr1i @ H1 @ gri1
        gri1 = grii @ H1 @ gri1
      
    grii = np.linalg.inv(E - H0 - H1@grii@H1.H - SigmaR)
    gr1i = gr1i @ H1.H @ grii
    gr11 = gr11 + gr1i @ H1 @ gri1
    gri1 = grii @ H1 @ gri1
#     G0x, G0y = np.concatenate([gr11,gr1i],axis=1), np.concatenate([gri1,grii],axis=1)
#     G0 = np.concatenate([G0x,G0y],axis=0)
#     G = np.linalg.inv(np.linalg.inv(G0)-Sigma_total)
#     gr1i = np.matrix(G[0:N,N:2*N])
    
#     print(np.imag(gr1i))
    
    didv = np.trace(GammaL@gr1i@GammaR@gr1i.H)
#     didv2 = np.trace(GammaL@gr11@GammaL@gr11.H)

    return np.real(didv),np.imag(didv)

