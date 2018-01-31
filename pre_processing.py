"""
BSD 3-Clause License

Copyright (c) 2017, Florian Engels
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy 
from numpy import *

class ProcessingSequence:
    def __init__(self,
                 mimo_demodulation,
                 array_preprocessing,
                 fourier_transformations,
                 noise_power_estimation,
                 peak_finding,
                 Tsl_r,Tsl_v,Tsl_a,Tsl_e,
                 scaling,
                 mimo_post_Fourier=True):
        self.mimo_post_Fourier = mimo_post_Fourier
        self.mimo_demodulation = mimo_demodulation
        self.array_preprocessing = array_preprocessing
        self.fourier_transformations = fourier_transformations
        self.noise_power_estimation = noise_power_estimation
        self.sidelobe_thresholding = SidelobeThresholding(Tsl_r,Tsl_v,Tsl_a,Tsl_e)
        self.peak_finding = peak_finding
        self.peak_finding.set_validate_peaks([self.noise_power_estimation.validate_peaks,
                                              self.sidelobe_thresholding.validate_peaks,
                                              self.mimo_demodulation.validate_peaks])                                
        self.Lf = self.fourier_transformations.Lf
        self.Mf = self.fourier_transformations.Mf
        self.Nf = self.fourier_transformations.Nf
        self.Of = self.fourier_transformations.Of
        self.w_r = self.fourier_transformations.w_r
        self.w_v = self.fourier_transformations.w_v
        self.w_a = self.fourier_transformations.w_a
        self.w_e = self.fourier_transformations.w_e
        Ls,Ms,Ns,Os = self.w_r.shape[0],self.w_v.shape[0],self.w_a.shape[0],self.w_e.shape[0]

        self.L,self.M,self.N,self.O = self.peak_finding.L,self.peak_finding.M,self.peak_finding.N,self.peak_finding.O       
        self.scaling = scaling
        self.ura_channels = self.array_preprocessing.ura_channels

    def __call__(self,x):
        print("Fast-time preprocessing")
        print("    Transpose")
        x = swapaxes(x,0,1)
        print("    DFT fast time")
        X = self.fourier_transformations.dft_fast_time(x)[0:(self.fourier_transformations.Lf//2),:,:]
        print("Preprocessing per range cell")
        if not self.mimo_post_Fourier:
            print("    MIMO Demodulation")
            X = self.mimo_demodulation.pre_fourier(X)
            print("    DFT slow time")
            X = self.fourier_transformations.dft_slow_time(X)[:,:,self.ura_channels] 
        else:    
            print("    Doppler DFTs")
            X = self.fourier_transformations.dft_slow_time(X)
            print("    MIMO demodulation")
            X = self.mimo_demodulation.post_fourier(X)[:,:,self.ura_channels]
        print("    Array preprocessing")
        X = self.array_preprocessing(X)
        print("    DFT y-channels")
        X = self.fourier_transformations.dft_y_channels(X)
        print("    DFT z-channels")
        X = self.fourier_transformations.dft_z_channels(X)

        X *= self.fourier_transformations.scale_dft_noise
        X *= self.scaling

        print("    Power spectrum")
        P = 20*log10(abs(X))
        print("    Noise power estimation")
        self.noise_power_estimation(P) 

        print("    4D Peak Finding")
        peak_list_4d = self.peak_finding(X,P)
        
        return X,P,self.noise_power_estimation.P_noise,peak_list_4d

class SlowTimeMimoDemodulation:
    def __init__(self,phase_codes,phase_increments=None):
        self.set_phase_codes(phase_codes)
        if phase_increments is not None:
            self.phase_increments = phase_increments

    def set_phase_codes(self,phase_codes):
        self.phase_codes = phase_codes
        self.Nt = self.phase_codes.shape[1]
        self.M = self.phase_codes.shape[0]
        self._determine_blocked_frequencies()

    def _determine_blocked_frequencies(self):
        self.fv_b = empty(0)
        A,S,Pe = exp(1j*self.phase_codes), abs(eye(self.Nt)-1),zeros(self.M+2)
        for nt in arange(self.Nt):
            P = abs(fft.fft(dot(A*outer(A[:,nt],ones(self.Nt)).conj(),S[:,nt])))**2
            Pe[1:-1],Pe[0],Pe[-1] = P,P[-1],P[0]
            peak_mask = ((diff(sign(diff(Pe,1)),1) < 0) & (P > (P.max()*10**(-20/10))))
            self.fv_b = append(self.fv_b,2*pi*arange(self.M)[peak_mask]/float(self.M))
        self.fv_b = unique(self.fv_b)

    def validate_peaks(self,P,lp,mp,np,op,
                       sz_fr,sz_fv,sz_fa,sz_fe,
                       fr_a=0,fv_a=0,fa_a=0,fe_a=0):
        valid_mask = ones(lp.shape,bool)
        Lr,Mr,Nr,Or = P.shape
        P = reshape(P,(Lr,Mr,Nr*Or))
        Pm=P.max(axis=2)
        for fv_b in self.fv_b:
            mb = numpy.round((( fv_a + sz_fv*mp + fv_b) % (2*pi))/sz_fv-fv_a).astype(int)
            mb[mb<0],mb[mb>=Mr] = mp[mb<0],mp[mb>=Mr]
            valid_mask &= Pm[lp,mp] >= Pm[lp,mb]
        return lp[valid_mask],mp[valid_mask],np[valid_mask],op[valid_mask]

    #Demodulation with arbitrary phase sequence 
    def pre_fourier(self,x):
        Ls,Ms,Nr = x.shape
        x_mimo,A = zeros((Ls,Ms,(Nr*self.Nt)),complex_),exp(-1j*self.phase_codes)
        for nt in arange(self.Nt):
            rc_nt = x*reshape(kronv(kronv(ones(Ls),A[:,nt]),ones(Nr)),(Ls,Ms,Nr))
            x_mimo[:,:,nt*Nr:(nt+1)*Nr] = rc_nt
        return x_mimo 

    def post_fourier(self,x):
        Ls,Ms,Nr = x.shape
        x_mimo = zeros((Ls,Ms,(Nr*self.Nt)),complex_)
        for nt in arange(self.Nt):
            shift = np.round((Ms/(2*pi))*self.phase_increments[nt]).astype(int)
            x_mimo[:,:,nt*Nr:(nt+1)*Nr] = roll(x,-shift,axis=1)
        return x_mimo 

class ArrayPreprocessing:
    def __init__(self,p_channels,dy_ura,dz_ura,ura_channels,sub_channels,Ns,Os,PC,C):
        self.Nc,self.Nvc = PC.shape
        self.PC,self.C = PC,C 
        self.Ns,self.Os = Ns,Os
        p_ura = dot((self.PC/sum(self.PC,axis=0)).T,p_channels[ura_channels])
        self.ns,self.os = self._calculate_ura_mapping(p_ura,dy_ura,dz_ura)
        self.p_channels = p_channels
        self.dy_ura,self.dz_ura = dy_ura,dz_ura
        self.ura_channels,self.sub_channels = ura_channels,sub_channels

    def __call__(self,x):
        Lr,Mr,Nv = x.shape
        x = self.map_on_ura(dot(dot(reshape(x,(Lr*Mr,self.Nc)),self.C),self.PC))
        return reshape(x,(Lr,Mr,self.Ns,self.Os))

    def _calculate_ura_mapping(self,p_ura,dy_ura,dz_ura):
        """ 
        Calculate indices for mapping channel positions to URA grid. 
        """
        
        if dy_ura is None:
            dy_ura=abs(diff(unique(sort(p_ura[:,1])))).min()
        if dz_ura is None:
            dz_ura=abs(diff(unique(sort(p_ura[:,2])))).min()
        #Regular grid indices 
        ns=np.round((p_ura[:,1]-p_ura[:,1].min())/dy_ura).astype(int)
        os=np.round((p_ura[:,2]-p_ura[:,2].min())/dz_ura).astype(int)
        return ns,os

    def map_on_ura(self,x):
        """ 
        Map channel positions to URA grid based on precalculated indices. 

        """
        x_ura = zeros((x.shape[0],self.Ns,self.Os),complex_)
        x_ura[:,self.ns,self.os] = x
        return x_ura


class FourierTransformations:
    def __init__(self,
                 Lf,Mf,Nf,Of,
                 w_r,w_v,w_a,w_e):

        self.Lf,self.Mf,self.Nf,self.Of = Lf,Mf,Nf,Of
        self.w_r,self.w_v,self.w_a,self.w_e = w_r,w_v,w_a,w_e
        self.Ls,self.Ms,self.Ns,self.Os = w_r.shape[0],w_v.shape[0],w_a.shape[0],w_e.shape[0]
        self.w = reshape(kronv(kronv(self.w_r,self.w_v),self.w_a),(self.Ls,self.Ms,self.Ns))
        self.output_scale = 1.
        self.output_scale_fast_time = 1.
        self.output_scale_slow_time = 1.
        self.output_scale_y_channels = 1.
        self.output_scale_z_channels = 1.
        self.scale_dft_noise = 1/sqrt(sum(w_r**2)*sum(w_v**2)*sum(w_a**2)*sum(w_e**2))

    def __call__(self,x):
        return fft.fftn(self.w*x,(self.Lf,self.Mf,self.Nf))/self.output_scale
    def dft_fast_time(self,x):
        Lr,Mr,Nr = x.shape
        w = reshape(kronv(kronv(self.w_r,ones(Mr)),ones(Nr)),(self.Ls,Mr,Nr))
        return fft.fft(w*x,self.Lf,axis=0)/self.output_scale_fast_time
    def dft_slow_time(self,x):
        Lr,Mr,Nr = x.shape
        w = reshape(kronv(kronv(ones(Lr),self.w_v),ones(Nr)),(Lr,self.Ms,Nr))
        return fft.fft(w*x,self.Mf,axis=1)/self.output_scale_slow_time
    def dft_y_channels(self,x):
        Lr,Mr,Ns,Os = x.shape
        w = reshape(kronv(kronv(kronv(ones(Lr),ones(Mr)),self.w_a),ones(Os)),(Lr,Mr,self.Ns,Os))
        return fft.fft(w*x,self.Nf,axis=2)/self.output_scale_y_channels
    def dft_z_channels(self,x):
        Lr,Mr,Nr,Os = x.shape
        w = reshape(kronv(kronv(kronv(ones(Lr),ones(Mr)),ones(Nr)),self.w_e),(Lr,Mr,Nr,Os))
        return fft.fft(w*x,self.Of,axis=3)/self.output_scale_z_channels

class NoisePowerEstimation:
    def __init__(self,order,T_power=12):
        self.order = order
        self.T_power=T_power

    def __call__(self,P):
        self.P_noise = zeros(P.shape)
        P_3d = sort(P,axis=1)[:,self.order,:,:]
        for m in arange(P.shape[1]):
            self.P_noise[:,m,:,:] = P_3d

    def validate_peaks(self,P,lp,mp,np,op,
                       sz_fr,sz_fv,sz_fa,sz_fe,
                       fr_a=0,fv_a=0,fa_a=0,fe_=0):
        valid_mask = P[lp,mp,np,op] >= (self.P_noise[lp,mp,np,op] + self.T_power)
        return lp[valid_mask],mp[valid_mask],np[valid_mask],op[valid_mask]

class SidelobeThresholding:
    def __init__(self,Tsl_r=100,Tsl_v=100,Tsl_a=100,Tsl_e=100):
        self.Tsl_r,self.Tsl_v,self.Tsl_a,self.Tsl_e = Tsl_r,Tsl_v,Tsl_a,Tsl_e

    def validate_peaks(self,P,lp,mp,np,op,
                       sz_fr,sz_fv,sz_fa,sz_fe,
                       fr_a=0,fv_a=0,fa_a=0,fe_a=0):
        valid_mask =  P[lp,mp,np,op] >= (P[:,mp,np,op].max(axis=0) - self.Tsl_r)
        valid_mask &= P[lp,mp,np,op] >= (P[lp,:,np,op].max(axis=1) - self.Tsl_v)
        valid_mask &= P[lp,mp,np,op] >= (P[lp,mp,:,op].max(axis=1) - self.Tsl_a)
        valid_mask &= P[lp,mp,np,op] >= (P[lp,mp,np,:].max(axis=1) - self.Tsl_e)
        return lp[valid_mask],mp[valid_mask],np[valid_mask],op[valid_mask]

class PeakFinding:
    def __init__(self,
                 sz_fr,sz_fv,sz_fa,sz_fe,
                 L=5,M=5,N=5,O=5,
                 fr_a=0,fv_a=0,fa_a=0,fe_a=0):
        self.L,self.M,self.N,self.O = L,M,N,O
        self.lm,self.mm,self.nm,self.om = self.L//2,self.M//2,self.N//2,self.O//2
        self.sz_fr,self.sz_fv,self.sz_fa,self.sz_fe = sz_fr,sz_fv,sz_fa,sz_fe
        self.validate_peaks = []
        self.fr_a,self.fv_a,self.fa_a,self.fe_a = fr_a,fv_a,fa_a,fe_a

    def set_validate_peaks(self,validate_peaks):
        self.validate_peaks = validate_peaks

    def __call__(self,X,P,Y=None):
        Lr,Mr,Nr,Or  = X.shape
        wrap_data = array([(self.fr_a + self.sz_fr*Lr) == (2*pi),
                           (self.fv_a + self.sz_fv*Mr) == (2*pi),
                           (self.fa_a + self.sz_fa*Nr) == (2*pi),
                           ((self.fe_a + self.sz_fe*Or) == (2*pi) and (Or > 2))])

        Pe = extend_data(P,1,1,1,1,wrap_data,-500)
        #Peak mask
        peaks_r = (diff(sign(diff(Pe,1,axis=0)),1,axis=0) < 0)
        peaks_v = (diff(sign(diff(Pe,1,axis=1)),1,axis=1) < 0)
        peaks_a = (diff(sign(diff(Pe,1,axis=2)),1,axis=2) < 0)
        peaks_e = (diff(sign(diff(Pe,1,axis=3)),1,axis=3) < 0)
        peaks_4d = (peaks_r[:,1:-1,1:-1,1:-1] &
                    peaks_v[1:-1,:,1:-1,1:-1] & 
                    peaks_a[1:-1,1:-1,:,1:-1] & 
                    peaks_e[1:-1,1:-1,1:-1,:])

        #Peak indices
        lp = reshape(kronv(kronv(kronv(arange(Lr),ones(Mr)),ones(Nr)),ones(Or)),(Lr,Mr,Nr,Or))[peaks_4d].astype(int)
        mp = reshape(kronv(kronv(kronv(ones(Lr),arange(Mr)),ones(Nr)),ones(Or)),(Lr,Mr,Nr,Or))[peaks_4d].astype(int)
        np = reshape(kronv(kronv(kronv(ones(Lr),ones(Mr)),arange(Nr)),ones(Or)),(Lr,Mr,Nr,Or))[peaks_4d].astype(int)
        op = reshape(kronv(kronv(kronv(ones(Lr),ones(Mr)),ones(Nr)),arange(Or)),(Lr,Mr,Nr,Or))[peaks_4d].astype(int)

        for vp in self.validate_peaks:
            lp,mp,np,op = vp(P,lp,mp,np,op,
                             self.sz_fr,self.sz_fv,self.sz_fa,self.sz_fe,
                             self.fr_a,self.fv_a,self.fa_a,self.fe_a)
        
        K = lp.shape[0]
        fr_ap,fv_ap,fa_ap,fe_ap = zeros(K),zeros(K),zeros(K),zeros(K)

        Xp = zeros((K,self.L,self.M,self.N,self.O),complex_)
        Xe = extend_data(X,self.L//2,self.M//2,self.N//2,self.O//2,wrap_data)

        if Y is not None:
            Yp = zeros((K,self.L,self.M,Y.shape[2]),complex_)
            Ye = extend_data(Y,self.L//2,self.M//2,self.N//2,self.O//2,wrap_data)
        else:
            Yp = None

        for i,(l_pc,m_pc,n_pc,o_pc) in enumerate(zip(lp,mp,np,op)):
           l_ab = l_pc + arange(-(self.L//2),self.L//2+1) 
           m_ab = m_pc + arange(-(self.M//2),self.M//2+1) 
           n_ab = n_pc + arange(-(self.N//2),self.N//2+1) 
           o_ab = o_pc + arange(-(self.O//2),self.O//2+1) 
           fr_ap[i] = self.fr_a + self.sz_fr*l_ab[0] 
           fv_ap[i] = self.fv_a + self.sz_fv*m_ab[0]
           fa_ap[i] = self.fa_a + self.sz_fa*n_ab[0]
           fe_ap[i] = self.fe_a + self.sz_fe*o_ab[0]

           Xp[i,:] = (((Xe[l_ab+self.L//2,:,:,:])[:,m_ab+self.M//2,:,:])[:,:,n_ab+self.N//2,:])[:,:,:,o_ab+self.O//2]
           if Y != None:
               Yp[i,:] = ((Ye[l_ab+self.L//2,:,:])[:,m_ab+self.M//2,:])[:,:,:]

        fp = {"Range"           : fr_ap,
              "Velocity"        : fv_ap,
              "Azimuth angle"   : fa_ap,
              "Elevation angle" : fe_ap}
        #4D peak list
        peak_list_4d = {"Peak neighborhood" : Xp,
                        "MIMO channels"     : Yp,
                        "Peak frequencies"  : fp}

        return peak_list_4d

def extend_data(x,le,me,ne,oe,wrap=array([True,True,True,True]),min_value=0):
    L,M,N,O = x.shape
    xe = min_value*ones((L+2*le,M+2*me,N+2*ne,O+2*oe),dtype=x.dtype)
    xe[le:-le,me:-me,ne:-ne,oe:-oe] = x
    if wrap[0]:
        xe[:le,me:-me,ne:-ne,oe:-oe] = x[-le:,:,:,:]
        xe[-le:,me:-me,ne:-ne,oe:-oe] = x[:le,:,:,:]
    if wrap[1]:
        xe[le:-le,:me,ne:-ne,oe:-oe] = x[:,-me:,:,:]
        xe[le:-le,-me:,ne:-ne,oe:-oe] = x[:,:me,:,:]
    if wrap[2]:
        xe[le:-le,me:-me,:ne,oe:-oe] = x[:,:,-ne:,:]
        xe[le:-le,me:-me,-ne:,oe:-oe] = x[:,:,:ne,:]
    if wrap[3]:
        xe[le:-le,me:-me,ne:-ne,:oe] = x[:,:,:,-oe:]
        xe[le:-le,me:-me,ne:-ne,-oe:] = x[:,:,:,:oe]

    return xe

def kronv(v1,v2):
    return outer(v1,v2).ravel()

