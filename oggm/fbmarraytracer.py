"""
fbmtracer.py
Author: Samuel Kachuck
Date: Sep 1, 2020

Provides the tracer particle class that contains fiber bundles for calving the
ssa1d class, along with a collection of random strength distributions and state
variable functions.
"""
from __future__ import division
import numpy as np
from util import *

class FBMFullTracer(object):
    def __init__(self, model, xsep0, Nf=10, dist=None, compState=None,
    stepState=None, xsep=None, **kwargs):
        """
        A container for tracer particles in a 1D fenics ice dynamics model.

        The tracers advect with the fluid velocity, but do not affect the flow
        until the fiber bundle models they carry break, at which point calving
        is triggered in the ice flow model.
        There are two kinds of state variables: those that are integrated
        through the simulation, and those that only depend on the instantaneous
        configuration of the ice. For the former, provide a function,
        stepState, that is used to intergate the state variable when the
        particles advect. For the latter, provide a function that computes the
        state variable when checking for calving. Only one may be specified.

        Parameters
        ----------
        model : a FluxBasedModel whose flowlines to populate with fiber bundles
        xsep0 : the initial separation for the fiber bundles
        Nf    : The number of fibers per bundle
        dist : a function that returns a random threshold
        compState(x, ssaModel) : a function that computes the path-independent state 
            variable from the ssaModel at locations x, default None. If both
            compState and stepState are None, compState is defined as
            strain_thresh.
        stepState(x, ssaModel) : a function that computes a discrete
            time-derivative of the state variable, for integrating, default
            None
        xsep : the separation of particles when added to the system

        Data
        ----
        N : number of particles
        x : location of particles
        s : threshold of particles
        state : the state of each particle

        Methods
        -------
        advect_particles
        add_particle
        remove_particle
        check_calving
        calve
        """
        # Properties of tracers
        self.x_bundles = []
        self.n_bundles = []
        self.f_bundles = []
        
        for fl in model.fls:
            if fl.length_m > 0:
                xfl = np.arange(0, fl.length_m, xsep0)
                self.x_bundles.append(xfl)
                self.n_bundles.append(len(xfl))
                self.f_bundles.append(np.zeros_like(xfl))
            else:
                self.x_bundles.append([0])
                self.n_bundles.append(1)
                self.f_bundles.append(0)

        self.xsep = xsep or xsep0
#        assert compState is None or stepState is None, 'Cannot specify both'
#        self.compState = compState
#        self.stepState = stepState
#        if compState is None and stepState is None:
#            self.compState = strain_thresh
#
        # Fiber bundles 
#        self.dist = dist or strict_dist()
        # Number of fibers per bundle
        self.Nf = int(Nf)
#        # Construct the fibers - random thresholds
#        self.xcs = self.dist((self.N, self.Nf))
#        # Force on each fiber bundle
#        self.F = np.zeros(self.N)
#        # Broken status of each fiber in each tracer
        # Extension of each bundle (ELS) is self.F/np.sum(self.ss, axis=1)
        #self.fiber_states = []
        #for fl_id, fl in enumerate(model.fls):
        #    self.fiber_states.append(np.zeros((self.n_bundles[fl_id], self.Nf


        self.listform=False

    def _toList(self):
        if self.listform: return
        self.x_bundles = [xfl.tolist() for xfl in self.x_bundles]
        self.f_bundles = [ffl.tolist() for ffl in self.f_bundles]
#        self.xcs = self.xcs.tolist()
#        self.ss = self.ss.tolist()
#        self.F = self.F.tolist()
        self.listform = True

    def _toArr(self):
        if not self.listform: return
        self.x_bundles = [np.asarray(xfl) for xfl in self.x_bundles]
        self.f_bundles = [np.asarray(ffl) for ffl in self.f_bundles]
#        self.xcs = np.asarray(self.xcs) 
#        self.ss = np.asarray(self.ss)
#        self.F = np.asarray(self.F)
        self.listform = False

#    @property
#    def fiber_extension(self):
#        return self.F/np.sum(self.ss, axis=1)
#    @property
#    def exceeded_threshold(self):
#        return self.xcs <= self.fiber_extension[:,None]
#
#    @property
#    def active_tracers(self):
#        broken = np.where(np.logical_and(self.exceeded_threshold, self.ss))
#        return np.unique(broken[0])
#
#    @property
#    def damage(self):
#        return 1-np.sum(self.ss,axis=1)/self.Nf
#
#    @property
#    def data(self):
#        return np.vstack([self.x, self.damage])
#
#    def force(self, F):
#        #print('Forcing with {}'.format(F))
#        self.F = F
#        # Find tracers with broken fibers 
#        for i in self.active_tracers: 
#            while any(self.ss[i]) and any(self.exceeded_threshold[i][self.ss[i]]):
#                j = np.argwhere(self.exceeded_threshold[i]*self.ss[i])[0][0]
#                #print('Breaking {} {}'.format(i,j))
#                self.ss[i, j] = False
    def add_tracer(self, xp, fl_id, state=0):
        """
        Introduce a new particle at location xp with threshold drawn from
        self.dist.
        """
        self._toList()
   
        i = np.searchsorted(self.x_bundles[fl_id], xp)
        self.x_bundles[fl_id].insert(i,xp)
        self.f_bundles[fl_id].insert(i,state)
#        self.xcs.insert(i, self.dist(self.Nf))
#        self.F.insert(i,state)
#        self.ss.insert(i,np.ones(self.Nf,dtype=bool))
        self.n_bundles[fl_id] += 1
        self._toArr()

    def remove_tracer(self, i, fl_id):
        """
        Revmoce ith particle from the flow, used in calving.
        """
        self._toList()
        self.x_bundles[fl_id].pop(i)
        self.f_bundles[fl_id].pop(i)
#        self.xcs.pop(i)
#        self.F.pop(i)
#        self.ss.pop(i)
        self.n_bundles[fl_id] -= 1
        self._toArr()

    def interp_to_particles(self, fl_id, fl, yfl):
        """Interpolate array yfl on flowline fl to particle positions.
        """
        return np.interp(self.x_bundles[fl_id], fl.dis_on_line*fl.dx_meter, yfl)

    def advect_particles(self, model, dt):
        """
        Advect particles with the flow represented by vector coefficients
        Ufunc. Drop in a new particle if required.
        """
        self._toArr()
        # interpolate ice-velocities to particle positions
        #U = np.array([ssaModel.U(x) for x in self.x])
        for fl_id, fl in enumerate(model.fls): 
            u_on_fl = np.nan_to_num(model.flux_stag[fl_id]/(model.section_stag[fl_id]+1e-6))
            # Staggered grid has an additional grid point at end to remove
            u_on_fl = u_on_fl[:-1] 
            u_at_tracers = self.interp_to_particles(fl_id, fl, u_on_fl) 

            strain_rate_on_fl = np.gradient(u_on_fl, fl.dx_meter)
            strain_rate_at_tracers = self.interp_to_particles(fl_id, fl, strain_rate_on_fl)

            self.x_bundles[fl_id] += u_at_tracers*dt
            self.f_bundles[fl_id] += strain_rate_at_tracers*dt

            # remove particles advected beyond the front.
            while np.any(self.x_bundles[fl_id] > fl.length_m) and self.n_bundles[fl_id]>1:
                self.remove_tracer(-1, fl_id)

#
#        self.force(F)
#
        # Drop new particles in at intervals of self.xsep
        for fl_id, fl in enumerate(model.fls):
            if self.x_bundles[fl_id][0] > self.xsep:
                self.add_tracer(0, fl_id)

#    def check_calving(self):
#        """Check if any FBMs have exceeded their thresholds.
#        """
#        if all(np.sum(self.ss, axis=1)):
#            return False
#        else:
#            return True
#
#    def calve(self, i=None, x=None):
#        """Remove broken tracer and tracers connected to the front.
#        """
#        if x is not None:
#            try:
#                i=np.argwhere(self.x > x)[0][0]
#                print('Removing particles beyond Lmax')
#            except IndexError:
#                print('No particles beyond Lmax')
#                return x
#            
#        elif i is None:
#            assert self.check_calving(), 'No index given, none to break'
#            i=np.argwhere(np.sum(self.ss,axis=1)==0)[0][0]
#            #print(np.mean(self.xcs[i])) 
#        xc = self.x[i]
#        j = self.N - 1
#        while j >= i:
#            self.remove_tracer(j)
#            j-=1
#                
#        return xc
#
#
