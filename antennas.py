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

class Isotropic(object):
    def __init__(self,
                 p_c=array([0,0,0]),
                 T=eye(3)):
        self.rotate(T)
        self.p_c = p_c

    def translate(self,p):
        self.p_c = p

    def rotate(self,T):
        self.T = T
        self.x = dot(self.T,array([1,0,0]))
        self.y = dot(self.T,array([0,1,0]))
        self.z = dot(self.T,array([0,0,1]))

    def far_field(self,d,kappa):
        return exp(2j*pi*kappa*dot(self.p_c,d))

class Patch(Isotropic):        
    def __init__(self,
                 p_c=array([0,0,0]),
                 T=eye(3),
                 l_eq_wl=0.39):
        Isotropic.__init__(self,p_c,T)
        self.l_eq_wl = l_eq_wl

    def far_field(self,d,kappa):
        l_eq = self.l_eq_wl/kappa
        g_y = cos(0.5*l_eq*2*pi*kappa*dot(self.y,d))
        g_z = cos(0.5*l_eq*2*pi*kappa*dot(self.z,d))

        #return g_y*g_z*exp(2j*pi*kappa*dot(self.p_c,d))
        return g_y*g_z*Isotropic.far_field(self,d,kappa)

class RectangularArray:
   def __init__(self,
                p_c,
                p_grid_h,
                p_grid_v,
                gain_db,
                wl,
                element,
                grid_phi=None,
                P_phi=None,
                grid_theta=None,
                P_theta=None,
                T=eye(3)):
       self.element = element
       self.element.translate(array([0,0,0]))

       self.p_c = p_c
       self.N_h = p_grid_h.shape[0]
       self.N_v = p_grid_v.shape[0]
       self.p_array_0 = (kron(ones((self.N_h,1)),outer(p_grid_v,array([0,0,1]))) + 
                         kron(outer(p_grid_h,array([0,1,0])),ones((self.N_v,1)))).T
       self.rotate(T)
       
       self.gain = 10**(gain_db/20.) 

       if (grid_phi is None) or (P_phi is None):
           self.w_h = ones(self.N_h)/self.N_h
       else:
           self.w_h = self.calc_weights(p_grid_h,grid_phi,P_phi,wl,"horizontal")
       if (grid_theta is None) or (P_theta is None):
           self.w_v = ones(self.N_v)/self.N_v   
       else:
           self.w_v = self.calc_weights(p_grid_v,grid_theta,P_theta,wl,"vertical")

       self.w = outer(self.w_h,self.w_v).ravel()
       self.wl_design = wl

   def calc_weights(self,pos,grid,P,wl,direction,N_int=1025):
       N = pos.shape[0]
       d_in = zeros((3,N_int))
       if direction == "horizontal":
           grid_int = linspace(-pi*0.5,pi*0.5,N_int)
           d_in[1,:] = sin(grid_int)
       elif direction == "vertical":
           grid_int = linspace(-pi*0.5,pi*0.5,N_int)#TODO: keep this?
           d_in[0,:] = sin(grid_int+pi*0.5)
           d_in[2,:] = cos(grid_int+pi*0.5)
       else:
           raise TypeError("direction has to be either horizontal or vertical.")

       sz = grid_int[1]-grid_int[0]

       P_int,B = -100.*ones(N_int),zeros(N_int,complex_)
       kappa=1./wl

       for i in arange(grid.shape[0]-1): 
           grid1,grid2 = sz*np.round(grid[i]/sz),sz*np.round(grid[i+1]/sz)
           i1,i2 = int(grid1/sz + N_int/2),int(grid2/sz + N_int/2 + 1) 
           P1,P2 = P[i],P[i+1]
           P_int[i1:(i2)] = ((P2-P1)/(grid2-grid1))*(arange(i2-i1)*sz) + P1

       B_patch = self.element.far_field(d_in,kappa)
       A = outer(ones(N),B_patch)*exp(2j*pi*kappa*outer(pos,sin(grid_int)))
       B.real = 10**(P_int/20.)
       w = real(dot(linalg.pinv(dot(A,A.conj().T)),dot(A,B.conj().T))) 
       w /= sum(w) 

       return w

   def rotate(self,T):
       self.element.rotate(T)
       self.T = T

       self.p_array =  dot(self.T,self.p_array_0) 
       self.p_array += kron(ones((self.N_h*self.N_v,1)),dot(self.T,self.p_c)).T

   #Acoording to Kildal2015, chapter 10.1     
   def far_field(self,p,kappa):
       return (self.gain*
               self.element.far_field(p,kappa)*
               sum(self.w*exp(2j*pi*kappa*dot(p,self.p_array))))

   def far_field_0(self,p,kappa):
       return (self.element.far_field(p,kappa)*
               sum(self.w*exp(2j*pi*kappa*dot(p,dot(self.T,self.p_array_0)))))


   def far_field_on_grid(self,grid_phi=linspace(-0.5*pi,0.5*pi,181),
                              grid_theta=linspace(0,pi,181)):
       Nphi,Ntheta = grid_phi.shape[0],grid_theta.shape[0]
       P = zeros((Ntheta,Nphi))
       kappa = 1/self.wl_design
       for n,phi in enumerate(grid_phi):
           for o,theta in enumerate(grid_theta):
               p = array([sin(theta)*cos(phi),sin(theta)*sin(phi),cos(theta)])
               x = self.far_field_0(p,kappa)
               P[o,n] = abs(x)**2

       return P,grid_theta,grid_phi
