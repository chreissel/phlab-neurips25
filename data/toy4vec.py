import numpy as np
import multiprocessing as mp
from pylorentz import Momentum4
from scipy.stats import beta
import multiprocessing as mp
from multiprocessing import Process, Pool
from dataclasses import dataclass
import math

@dataclass
class particle:
    mom: Momentum4
    randtheta: float
    z: float
    m1: float
    m2: float


class jet_data_generator(object):
    """    
    Input takes the following form. (massprior,quarkmass,nprong,nparticle)
    
    massprior : "signal" or "background" (signal to use Gaussian prior, backgroound for uniform)
    quarkmass : mass of the quark in the showering
    nprong : number of particles after hard splitting
    nparticle: total number of particles after showering 
    
    Then use generate_dataset(N) to generate N number of events  
    
    """
    def __init__(self, massprior, nprong, nparticle, doFixP,nrandparticle=0,       doMultiprocess=False, ncore = 0):
        super(jet_data_generator, self).__init__()
        self.massprior = massprior
        self.nprong    = nprong
        self.nparticle = nparticle
        self.nrandparticle = nrandparticle
        self.zsoft = []
        self.zhard = []
        self.z = []
        self.randtheta = []
        self.doFixP = doFixP
        self.doMultiprocess = doMultiprocess
        self.ncore = ncore

    def reverse_insort(self, a, x, lo=0, hi=None):
        """Insert item x in list a, and keep it reverse-sorted assuming a
        is reverse-sorted. The key compared is the invariant mass of the 4-vector

        If x is already in a, insert it to the right of the rightmost x.

        Optional args lo (default 0) and hi (default len(a)) bound the
        slice of a to be searched.
        """
        if lo < 0:
            raise ValueError('lo must be non-negative')
        if hi is None:
            hi = len(a)
        while lo < hi:
            mid = (lo+hi)//2
            if (x.mom.m > a[mid].mom.m and x.mom.p > 1) or  (x.mom.p >  a[mid].mom.p and x.mom.p < 1): hi = mid
            else: lo = mid+1
        a.insert(lo, x)

    def rotation_matrix(self,axis, theta):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        axis = np.asarray(axis)
        axis = axis / math.sqrt(np.dot(axis, axis))
        a = math.cos(theta / 2.0)
        b, c, d = -axis * math.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

        
    def theta_to_eta(self,theta):
        if theta > np.pi:
            theta = 2*np.pi - theta
        return -np.log(np.tan(theta/2))

    def sintheta1(self,z,theta):
        return (-z/(1-z))*np.sin(theta)

    def mass(self,z,theta):
        sint1=self.sintheta1(z,theta)
        cost1=np.sqrt(1-sint1**2)
        p=z*np.cos(theta)+(1-z)*cost1
        return np.sqrt(1-p**2)

    def gamma(self,z,theta):
        return 1./self.mass(z,theta)

    def betaval(self,gamma):
        return np.sqrt(1-1/gamma**2)

    def sinthetaR(self,z,theta):
        gammavar=self.gamma(z,theta)
        K=gammavar*np.tan(theta)
        betavar=self.betaval(gammavar)
        return 2*K/(K**2+betavar)

    def sinthetaR2(self,z,theta):#more robust solution
        mom=(self.mass(z,theta)/2)
        return z*np.sin(theta)/mom
    
    def restmom(self,z,theta):
        sintR=self.sinthetaR(z,theta)
        gammavar=self.gamma(z,theta)
        betavar=self.betaval(gammavar)
        return z*betavar*np.sin(theta)/sintR

    def p2(self,iM1,iM2,iMM):
        """       
        #Phil, fix to ensure momentum is back to back and mass is perserved
        #sqrt(p^2+m1^2)+sqrt(p^2+m2^2)=mother.mom.m=> solve for p^2
        """
        return (iM1**4+iM2**4+iMM**4-2*(iM1*iM2)**2-2*(iM1*iMM)**2-2*(iM2*iMM)**2)/(2*iMM)**2

    def rotateTheta(self,idau,itheta):
        v1     = [idau.p_x,idau.p_y,idau.p_z]
        axis=[0,1,0]
        v1rot=np.dot(self.rotation_matrix(axis,itheta), v1)
        dau1_mom = Momentum4(idau.e,v1rot[0],v1rot[1],v1rot[2])
        return dau1_mom

    def rotatePhi(self,idau,itheta):
        v1     = [idau.p_x,idau.p_y,idau.p_z]
        axis=[0,0,1]
        v1rot=np.dot(self.rotation_matrix(axis,itheta), v1)
        dau1_mom = Momentum4(idau.e,v1rot[0],v1rot[1],v1rot[2])
        return dau1_mom

    def hardsplit(self, mother, nthsplit):
        #Hard splitting performed in the rest frame of the mother, rotated, and then lorentz boosted back
        #Hard splitting prior: Gaussian around pi/2,
        np.random.seed()
        randomdraw_theta = np.random.uniform(0.1,np.pi/2.-0.1)
        randomdraw_phi   = np.random.uniform(0,2*np.pi)
        dau1_m           = mother.mom.m/10.#np.random.uniform(mother.mom.m/16, mother.mom.m/2)
        dau2_m           = mother.mom.m/10.#np.random.uniform(mother.mom.m/16, mother.mom.m/2)
        dau1_theta = (np.pi/2 + randomdraw_theta)
        dau2_theta = (np.pi/2 - randomdraw_theta)        
        dau1_phi = mother.mom.phi + randomdraw_phi
        dau2_phi = mother.mom.phi + randomdraw_phi + np.pi
        dau1_phi %= (2*np.pi)
        dau2_phi %= (2*np.pi)
        #prep for 4-vector
        dau_p2   = self.p2(dau1_m,dau2_m,mother.mom.m)
        dau1_e   = np.sqrt(dau_p2+dau1_m**2)
        dau2_e   = np.sqrt(dau_p2+dau2_m**2)
        dau1_mom = Momentum4.e_m_eta_phi(dau1_e, dau1_m, self.theta_to_eta(dau1_theta), dau1_phi)
        dau2_mom = Momentum4.e_m_eta_phi(dau2_e, dau2_m, self.theta_to_eta(dau2_theta), dau2_phi)
        #print("0-eta:",dau1_mom.eta)
        dau1_mom = self.rotateTheta(dau1_mom,mother.mom.theta-np.pi/2)
        dau2_mom = self.rotateTheta(dau2_mom,mother.mom.theta-np.pi/2)
        #print("1-eta:",dau1_mom.eta,mother.mom.theta-np.pi/2)
        dau1_mom = self.rotatePhi(dau1_mom,mother.mom.phi)
        dau2_mom = self.rotatePhi(dau2_mom,mother.mom.phi)
        #print("1.4-eta:",dau1_mom.eta)
        dau1 = particle(mom=dau1_mom,randtheta=-1000,z=-1000,m1=-1000,m2=-1000)
        dau2 = particle(mom=dau2_mom,randtheta=-1000,z=-1000,m1=-1000,m2=-1000)
        #print(dau1.mom.p_t,dau2.mom.p_t,dau1.mom.phi,dau2.mom.phi,mother.mom.phi,"d1")
        #print("1.5-eta:",dau1.mom.eta)
        dau1.mom = dau1.mom.boost_particle(mother.mom)
        dau2.mom = dau2.mom.boost_particle(mother.mom)
        #print(dau1.mom.p_t,dau2.mom.p_t,dau1.mom.phi,dau2.mom.phi,"d2")
        #print("2-eta:",dau1.mom.eta,mother.mom.eta)
        self.randtheta.append(randomdraw_theta)
        self.zhard.append(np.min([dau1.mom.p_t, dau2.mom.p_t])/(dau1.mom.p_t+dau2.mom.p_t))
        self.z.append(np.min([dau1.mom.p_t, dau2.mom.p_t])/(dau1.mom.p_t+dau2.mom.p_t))
        return dau1, dau2, np.min([dau1.mom.p_t, dau2.mom.p_t])/(dau1.mom.p_t+dau2.mom.p_t), randomdraw_theta
    
    def draw_first_particle(self):
        np.random.seed()
        if self.massprior == "signal":
            m = np.random.normal(100, 20)
            p = np.random.uniform(400,1000)

        if self.massprior == "signal_data":
            m = np.random.normal(75, 22)
            p = np.random.uniform(400,1000)
            
        if self.massprior == "background":
            m = np.random.uniform(0,100)
            p= np.random.uniform(400,1000)
            
        #p = np.random.exponential(400)
        #delete later
        #if self.doFixP:
        #p = 400
        vec0 = Momentum4.m_eta_phi_p(m, 0, 0, p)
        part = particle(mom=vec0,randtheta=-1000,z=-1000,m1=-1000,m2=-1000)  
        return part

    def hard_decays(self):
        hardparticle_list = [self.draw_first_particle()]
        prong = 1
        zlist = []
        thetalist = []
        while prong < self.nprong:
            dau1, dau2, z, theta = self.hardsplit(hardparticle_list[0],prong)
            hardparticle_list.pop(0)
            self.reverse_insort(hardparticle_list, dau1)
            self.reverse_insort(hardparticle_list, dau2)
            zlist.append(z)
            thetalist.append(theta)
            prong += 1
            
        return hardparticle_list, zlist, thetalist
    
    def genshower(self,_):
        showered_list, zlist, thetalist = self.hard_decays()
        total_particle = len(showered_list)        
        #while total_particle < self.nparticle:
        #    if showered_list[0].mom.p < 1:
        #        break
        #    #print(self.softsplit(showered_list[0]))
        #    dau1, dau2, z, theta = self.softsplit(showered_list[0])
        #    if dau1.z == -1:
        #        break
        #    #print(dau1.mom,showered_list)
        #    showered_list.pop(0)
        #    self.reverse_insort(showered_list, dau1)
        #    self.reverse_insort(showered_list, dau2)
        #    
        #    zlist.append(z)
        #    thetalist.append(theta)
        #    total_particle +=1
        return total_particle, showered_list, zlist, thetalist

    def shower(self,_):
        i=0
        total_particle,showered_list,zlist, thetalist=self.genshower(i)
        while total_particle <  self.nparticle:
            total_particle,showered_list,zlist, thetalist=self.genshower(i)
        arr = []

        check = Momentum4(0,0,0,0)
        for j in range(self.nparticle):
            arr.append(showered_list[j].mom.p_t/500.)
            arr.append(showered_list[j].mom.eta)
            arr.append(showered_list[j].mom.phi)
            check += showered_list[j].mom

        for j in range(self.nrandparticle):
            randp_t = np.random.uniform()*400.
            randeta = np.random.normal()*0.15
            randphi = np.random.normal()*0.15
            arr.append(randp_t/500.)
            arr.append(randeta)
            arr.append(randphi)

        tmp_arr = np.reshape(arr,((self.nparticle+self.nrandparticle),3))
        indx    = tmp_arr.argsort(axis=0)[:,0]
        tmp_arr = tmp_arr[indx]
        arr=tmp_arr.flatten()
        return np.squeeze(np.array(arr)), np.squeeze(np.array(zlist)), np.squeeze(np.array(thetalist))


    def generate_dataset(self, nevent):
        data       = np.empty([nevent, 3*(self.nparticle+self.nrandparticle)], dtype=float)
        data_z     = np.empty([nevent, (self.nparticle)-1], dtype=float)
        data_theta = np.empty([nevent, (self.nparticle)-1], dtype=float)
        
        for i in range(nevent):
            if i % 1000 == 0:
                print("xevent :",i)
            arr, arr_z, arr_theta = self.shower(i)    
            data[i]  = np.squeeze(arr)
            data_z[i] = np.squeeze(arr_z)
            data_theta[i] = np.squeeze(arr_theta)
        
        return np.array(data), np.array(data_z), np.array(data_theta)
