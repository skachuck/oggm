"""
fbmarraytracer.py
Author: Samuel Kachuck
Date: Apr 22, 2021

Provides the tracer particle class that contains fiber bundles for calving the
FlowlineModel, along with a collection of random strength distributions and state
variable functions.
"""
from __future__ import division
import numpy as np
from oggm.utils import strict_dist

class FBMFullTracer(object):
    def __init__(self, model, xsep0=100, Nf=10, dist=None, xsep=None, **kwargs):
        """
        A container for tracer particles in the flowlines of OGGM FluxBasedModel.

        The tracers advect with the fluid velocity, but do not affect the flow
        until the fiber bundle models they carry break, at which point calving
        is triggered in the ice flow model.  

        Parameters
        ----------
        model   : a FluxBasedModel whose flowlines to populate with fiber bundles
        xsep0   : the initial separation for the fiber bundles (defauly 100 m)
        Nf      : The number of fibers per bundle (default 10)
        dist    : a function that returns a random threshold (default
                    strict_dist)
        xsep    : the separation of particles when added to the system

        Data
        ----
        n_bundles       : number of bundles in each flowline
        x_bundles       : locations of bundles in each flowline
        f_bundles       : force on each bundle in each flowline
        fiber_thresholds: Nf fiber strengths in each bundle in each flowline
        fiber_states    : Nf states (intact=True, broken=False) in each bundle
                            in each flowline

        Methods
        -------
        advect_particles
        add_particle
        remove_particle
        check_calving
        calve
        """


        self.dist = dist or strict_dist(lo=0,hi=1.0)
        self.Nf = int(Nf)

        # Properties of tracers
        self.x_bundles = []         # Locations of bundles along each flowline
        self.n_bundles = []         # Number of bundles along each flowline
        self.f_bundles = []         # Forces on bundles along each flowline
        # These are quantities for each individual fiber (Nf of them) within 
        # each bundle along each flowline.
        self.fiber_thresholds = []
        self.fiber_states = []
        
        # Construct the initial bundle list per flowline. There must be at
        # least one bundle in each flowline.
        for fl in model.fls:
            if fl.length_m > 0:
                xfl = np.arange(0, fl.length_m, xsep0)
                self.x_bundles.append(xfl)
                self.n_bundles.append(len(xfl))
                self.f_bundles.append(np.zeros_like(xfl))
                self.fiber_thresholds.append(self.dist(len(xfl), Nf))
                self.fiber_states.append(np.ones((len(xfl), Nf), dtype=bool))
            else:
                self.x_bundles.append(np.array([0.]))
                self.n_bundles.append(1)
                self.f_bundles.append(np.array([0.]))
                self.fiber_thresholds.append(self.dist((1, Nf)))
                self.fiber_states.append(np.ones((1, Nf), dtype=bool))

        self.xsep = xsep or xsep0
        self.listform=False

        # For debugging purposes.
        self._allow_breaking = True



    def _toList(self):
        if self.listform: return
        self.x_bundles = [xfl.tolist() for xfl in self.x_bundles]
        self.f_bundles = [ffl.tolist() for ffl in self.f_bundles]
        self.fiber_thresholds = [ffl.tolist() for ffl in self.fiber_thresholds]
        self.fiber_states = [ffl.tolist() for ffl in self.fiber_states]
        self.listform = True

    def _toArr(self):
        if not self.listform: return
        self.x_bundles = [np.asarray(xfl) for xfl in self.x_bundles]
        self.f_bundles = [np.asarray(ffl) for ffl in self.f_bundles]
        self.fiber_thresholds = [np.asarray(ffl) for ffl in self.fiber_thresholds]
        self.fiber_states = [np.asarray(ffl) for ffl in self.fiber_states]
        self.listform = False


    def bundle_extension(self, fl_id):
        """Compute the extension of each bundle in flowline fl_id."""
        return self.f_bundles[fl_id]/np.sum(self.fiber_states[fl_id], axis=1)

    def exceeded_threshold(self, fl_id):
        """Locate fibers in bundles along flowline fl_id that should break."""
        return self.fiber_thresholds[fl_id] <= self.bundle_extension(fl_id)[:,None]

    def active_bundles(self, fl_id):
        """Locate indices of bundles along fl_id with fibers that should break."""
        broken = np.where(np.logical_and(self.exceeded_threshold(fl_id), 
                        self.fiber_states[fl_id]))
        return np.unique(broken[0])

    def damage(self, fl_id):
        """Compute proportion of each bundles fibers that have broken."""
        return 1-np.sum(self.fiber_states[fl_id],axis=1)/self.Nf

    def pull_bundles_and_break_fibers(self, fl_id):
        """"""
        # Find bundles with broken fibers 
        for i in self.active_bundles(fl_id): 
            while (any(self.fiber_states[fl_id][i]) and
                    any(self.exceeded_threshold(fl_id)[i][self.fiber_states[fl_id][i]])):
                j = np.argwhere(self.exceeded_threshold(fl_id)[i]*self.fiber_states[fl_id][i])[0][0]
                self.fiber_states[fl_id][i, j] = False

    def add_tracer(self, xp, fl_id, state=0):
        """
        Introduce a new particle at location xp with threshold drawn from
        self.dist.
        """
        self._toList()
   
        i = np.searchsorted(self.x_bundles[fl_id], xp)
        self.x_bundles[fl_id].insert(i,xp)
        self.f_bundles[fl_id].insert(i,state)
        self.fiber_thresholds[fl_id].insert(i,self.dist(self.Nf))
        self.fiber_states[fl_id].insert(i,np.ones(self.Nf,dtype=bool))
        self.n_bundles[fl_id] += 1
        self._toArr()

    def remove_tracer(self, i, fl_id):
        """Revmoce ith particle from flowline fl_id, used in calving."""
        self._toList()
        self.x_bundles[fl_id].pop(i)
        self.f_bundles[fl_id].pop(i)
        self.fiber_thresholds[fl_id].pop(i)
        self.fiber_states[fl_id].pop(i)
        self.n_bundles[fl_id] -= 1
        self._toArr()

    def interp_to_particles(self, fl_id, fl, yfl):
        """Interpolate array yfl on flowline fl to particle positions."""
        return np.interp(self.x_bundles[fl_id], fl.dis_on_line*fl.dx_meter, yfl)

    def advect_particles(self, model, dt):
        """
        Advect bundles with the fluid velocity in model. 
        
        Drops in a new particle if required.
        """
        self._toArr()

        # Advect bundles along each flowline
        for fl_id, fl in enumerate(model.fls): 
            # Compute velocity and fill in values.
            u_on_fl = np.nan_to_num(model.flux_stag[fl_id]/(model.section_stag[fl_id]+1e-6))
            # Staggered grid has an additional grid point at end to remove
            u_on_fl = u_on_fl[:-1] 
            # Interpolate velocity to bundle locations
            u_at_tracers = self.interp_to_particles(fl_id, fl, u_on_fl) 

            # Compute the strain rate, integrated strain is used to force
            # bundles.
            strain_rate_on_fl = np.gradient(u_on_fl, fl.dx_meter)
            strain_rate_at_tracers = self.interp_to_particles(fl_id, fl, strain_rate_on_fl)
            # Update positions and integrated strains of bundles
            self.x_bundles[fl_id] += u_at_tracers*dt
            self.f_bundles[fl_id] += strain_rate_at_tracers*dt

            # remove particles advected beyond the front.
            # TODO transfer particles from tributaries, if applicable.
            while np.any(self.x_bundles[fl_id] > fl.length_m) and self.n_bundles[fl_id]>1:
                self.remove_tracer(-1, fl_id)
            # Break any fibers that need to be broken.
            self.pull_bundles_and_break_fibers(fl_id)

        # Drop new particles in at intervals of self.xsep
        for fl_id, fl in enumerate(model.fls):
            if self.x_bundles[fl_id][0] > self.xsep:
                self.add_tracer(0, fl_id)

    def calve(self, fl_id, x):
        """Remove bundles at and to the right of x along fl_id."""
        i = np.searchsorted(self.x_bundles[fl_id], x)-1

        j = self.n_bundles[fl_id] - 1
        while j >= i:
            self.remove_tracer(j, fl_id)
            j-=1

