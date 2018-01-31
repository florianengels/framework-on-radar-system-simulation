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
import numpy as np 
from numpy import *

class CoherentPulseTrain(object):
    """ 
    Coherent pulse train base class.

    Instance Variables 
    
    fc             -- carrier frequency of the radar transmit pulse
    tr             -- pulse repetition time
    ts             -- sampling time
    tss            -- pulse start to sampling delay 
    Ls,Ms          -- sample support in fast and slow time, respectively
    antennas_t,    -- transmit and receive antennas, respectively 
    antennas_r
    dy_ura,        -- spacing of uniform rectangular array (URA) grid 
    dz_ura            on which the actual antenna positions are mapped 
    ura_channels   -- list of channels to be mapped to an URA 
    sub_channels   -- list of channels which are used for sub-array 
                      processing, i.e. resolving potential ambiguities 
                      of the URA    
    Pt             -- transmit power in dBm
    Fn             -- noise figure 
    Bn             -- effective noise bandwidth
    loss           -- losses in the Rx and Tx path, e.g. feed-line losses
    gain_lna       -- Gain of the low noise amplifier (LNA) prior to sampling
    c              -- wave propagation velocity
    kappa          -- wavenumber
    """
    def __init__(self,
                 fc,tr,ts,tss,
                 Be,
                 Ls,Ms,
                 antennas_t,antennas_r,
                 Pt,Fn,Bn,loss,gain_lna):

        self.fc,self.tr,self.ts,self.tss,self.Ls,self.Ms = fc,tr,ts,tss,Ls,Ms
        self.Be = Be
        self.antennas_t,self.antennas_r = antennas_t,antennas_r
        self.Nt = len(self.antennas_t)
        self.Nr = len(self.antennas_r)
        self.Pt_dBm = Pt
        self.Pt= 10**((self.Pt_dBm+10*log10(1e-3))/10)
        self.loss = 10**(loss/10)
        self.gain_lna = gain_lna
        self.Const = (3e8/self.fc)**2/((4*pi)**3)
        self.c = 3e8
        self.kappa = self.fc/self.c
        self.Pn = self._calculate_noise_power(Fn,Bn)
        self.p_channels = self._calculate_channel_positions()


    def _calculate_noise_power(self,Fn,Bn):
        """ 
        Calculate thermal noise power over effective bandwidth. 

        Fn   -- Noise figure
        Bn   -- Noise bandwidth
        """
        kT0=10*log10(1.38e-23*290)
        self.Fn,self.Bn = Fn,Bn
        return 10**((kT0+Fn+10*log10(Bn))/10)

    def _calculate_channel_positions(self):
        """ 
        Calculate channel position. 
        For MIMO schemes this function is overriden to include virtual channels, i.e. Tx and Rx combinations.  
        """
        p_channels = zeros((self.Nr,3))
        for (nr,pr) in enumerate(self.antennas_r):
            p_channels[nr,:] = pr.p_c
        p_channels /= (self.c/self.fc)
        return p_channels

    def _sv_ft(self,dpl):
        """
        Derived classes shall implement the Calculate steering vector in fast-time dimension for a given path length. 
        
        pl      -- path length
        return  -- temporal steering vector in fast-time dimension
        """
        return exp(2j*pi*self.kappa*self.tr*dpl*arange(self.Ms))

    def _sv_st(self,dpl):
        """
        Calculate temporal steering vector in slow-time dimension for a given path length derivative. 
        
        dpl      -- temporal derivative of path length
        return   -- temporal steering vector in slow-time dimension
        """
        return exp(2j*pi*self.kappa*self.tr*dpl*arange(self.Ms))

    def _sv_tx(self,d):
        """
        Calculate spatial steering vector for transmit array for a given propagation direction. 
        
        d   -- propagation direction
        sv  -- spatial steering vector 
        """
        sv = zeros(len(self.antennas_t),complex_) 
        for i,antenna in enumerate(self.antennas_t):
            sv[i] = antenna.far_field(d,self.kappa)
        return sv 

    def _sv_rx(self,d):
        """
        Calculate spatial steering vector for receive array for a given propagation direction. 
        
        d   -- propagation direction
        sv  -- spatial steering vector 
        """
        sv = zeros(len(self.antennas_r),complex_) 
        for i,antenna in enumerate(self.antennas_r):
            sv[i] = antenna.far_field(d,self.kappa)
        return sv 

    def _combine_steering_vectors(self,sv_ft,sv_st,sv_tx,sv_rx):
        """
        Phased array version, i.e. uncoded superposition of tx antennas. 
        
        sv_ft       -- fast-time steering vector
        sv_st       -- slow-time steering vector
        sv_rx       -- steering vector rx-channels
        sv_tx       -- steering vector tx-antennas

        return   -- combined steering vector for fast time, slow time, rx channels, and tx antennas   
        """
        return dot(kronv(kronv(kronv(sv_st,sv_ft),sv_rx),ones(self.Nt)),sv_tx)

    def _scale_amplitude(self,pl):
        return (sqrt((3e8/self.fc)**2/((4*pi)**3)*self.Pt/self.loss))*exp(-2j*pi*self.kappa*pl)

    def __call__(self,
                 a_k, 
                 pl_k, dpl_k,
                 dod_k,doa_k):
        """
        Simulate radar baseband signal for multiple propagation paths.  
        
        a_k    -- Amplitudes   
        pl_k   -- Propagation path length
        dpl_k  -- Temporal derivative of path length
        dod_k  -- Direction of departure (DOD) 
        doa_k  -- Direction of arrival (DOA) 

        return -- Threedimensional baseband signal, i.e dimensions 
                  correspond to chirps, fast time, and rx channels  
        """
        self.x = zeros(self.Ls*self.Ms*self.Nr)

        a_k *= self._scale_amplitude(pl_k)

        for a,pl,dpl,dod,doa in zip(a_k,pl_k,dpl_k,dod_k.T,doa_k.T):
            self.x += real(a*self._combine_steering_vectors(self._sv_ft(pl),
                                                            self._sv_st(dpl),
                                                            self._sv_tx(dod),
                                                            self._sv_rx(doa)))
        
        self.x = reshape(self.x,(self.Ms,self.Ls,self.Nr))
        self.x += sqrt(self.Pn)*random.randn(self.Ms,self.Ls,self.Nr)/sqrt(2)
        self.x *= self.gain_lna
 
class SlowTimeMimoPhaseCoding(CoherentPulseTrain):
    """
    Slow-time MIMO scheme with phase coding per pulse. 

    Additional instance Variables 
    
    mimo_phase_codes       -- Phase codes in slow-time dimensions 
    mimo_phase_increments  -- Phase increment for linear phase variation
    """
    def __init__(self,
                 fc,
                 tr,ts,tss, 
                 Be,
                 Ls,Ms,
                 antennas_t,antennas_r,
                 Pt,Fn,Bn,loss,gain_lna,
                 mimo_phase_codes,
                 mimo_phase_increments=None):

        CoherentPulseTrain.__init__(self,
                                    fc,tr,ts,tss,
                                    Be,
                                    Ls,Ms,
                                    antennas_t,antennas_r,
                                    Pt,Fn,Bn,loss,gain_lna)
        self.mimo_phase_codes = mimo_phase_codes
        if mimo_phase_increments is not None:
            self.mimo_phase_increments = mimo_phase_increments


    def _calculate_channel_positions(self):
        p_channels = zeros((self.Nr*self.Nt,3))

        for (nt,pt) in enumerate(self.antennas_t):
            for (nr,pr) in enumerate(self.antennas_r):
                nv = nr + self.Nr*nt
                p_channels[nv,:] = pt.p_c + pr.p_c
        p_channels /= (self.c/self.fc)
        return p_channels

    def _combine_steering_vectors(self,sv_ft,sv_st,sv_tx,sv_rx):
        """
        MIMO version, i.e. slow-time-phase-coded superposition of tx antennas. 
        
        sv_ft       -- fast-time steering vector
        sv_st       -- slow-time steering vector
        sv_rx       -- steering vector rx-channels
        sv_tx       -- steering vector tx-antennas

        return   -- combined steering vector for fast time, slow time, and rx channels.   
        """
        sv_ts = dot((outer(sv_st,ones(self.Nt))*exp(1j*self.mimo_phase_codes)),sv_tx)
        return kronv(kronv(sv_ts,sv_ft),sv_rx)

class ChirpSequence(CoherentPulseTrain):
    """
    Chirp sequence modulation, i.e. coherent pulse train with linear frequency modulated pulses. 

    """
    def __init__(self,
                 fc,tr,ts,tss,        
                 Be,         
                 Ls,Ms,
                 antennas_t,antennas_r,
                 Pt,Fn,Bn,loss,gain_lna):

        CoherentPulseTrain.__init__(self,
                                    fc,tr,ts,tss,
                                    Be,
                                    Ls,Ms,
                                    antennas_t,antennas_r,
                                    Pt,Fn,Bn,loss,gain_lna)

    def _sv_ft(self,pl):
        """
        Calculate temporal steering vector in fast-time dimension for a given path length. 
        
        pl       -- path length
        return   -- temporal steering vector in fast-time dimension
        """
        return exp(2j*pi*(self.Be/(self.c*self.Ls))*pl*arange(self.Ls))

    def _if_filter(self,r):
        return 1.

    def _scale_amplitude(self,pl):
        return CoherentPulseTrain._scale_amplitude(self,pl)*self._if_filter(pl)

    def scale(self):
        return 1.


class ChirpSequenceMimoPhaseCoding(ChirpSequence,SlowTimeMimoPhaseCoding):
   """
   Chirp sequence modulation combined with slow-time MIMO phase coding.

   """
   def __init__(self,
                fc,tr,ts,tss,        
                Be,         
                Ls,Ms,
                antennas_t,antennas_r,
                Pt,Fn,Bn,loss,gain_lna,
                mimo_phase_codes,
                mimo_phase_increments=None):

       ChirpSequence.__init__(self,
                              fc,tr,ts,tss,
                              Be,
                              Ls,Ms,
                              antennas_t,antennas_r,
                              Pt,Fn,Bn,loss,gain_lna)
       SlowTimeMimoPhaseCoding.__init__(self,
                                        fc,tr,ts,tss,
                                        Be,
                                        Ls,Ms,
                                        antennas_t,antennas_r,
                                        Pt,Fn,Bn,loss,gain_lna,
                                        mimo_phase_codes,
                                        mimo_phase_increments)

def calc_target_parameters(rcs,p,v,p_s=array([0,0,0]),v_s=array([0,0,0])):
    K = p.shape[1]
    p -= outer(p_s,ones(K))
    r = sqrt(p[0,:]**2+p[1,:]**2+p[2,:]**2)
    d = p/r
    v -= outer(v_s,ones(K))
    vr = ((v[0,:]*d[0,:]) + (v[1,:]*d[1,:]) + (v[2,:]*d[2,:]))
    a = ((10**(rcs/20.0))/r)*exp(1j*0)
    return a,r,d,vr 

def kronv(v1,v2):
    return outer(v1,v2).ravel()

