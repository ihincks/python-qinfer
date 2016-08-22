#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# qhl.py: Models for quantum Hamiltonian (and generator) learning.
##
# © 2014 Chris Ferrie (csferrie@gmail.com) and
#        Christopher E. Granade (cgranade@gmail.com)
#     
# This file is a part of the Qinfer project.
# Licensed under the AGPL version 3.
##
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
##

## FEATURES ###################################################################

from __future__ import absolute_import
from __future__ import division # Ensures that a/b is always a float.
from future.utils import with_metaclass

## ALL ########################################################################

# We use __all__ to restrict what globals are visible to external modules.
__all__ = [
]

## IMPORTS ####################################################################

import numpy as np
import abc

from qinfer.abstract_model import FiniteOutcomeModel
from qinfer.domains import IntegerDomain

# Since the rest of QInfer does not require QuTiP,
# we need to import it in a way that we don't propagate exceptions if QuTiP
# is missing or is too early a version.
try:
    import qutip as qt
    from distutils.version import LooseVersion
    _qt_version = LooseVersion(qt.version.version)
    if _qt_version < LooseVersion('3.1'):
        qt = None
except ImportError:
    qt = None

## CLASSES ####################################################################

class QHLModel(with_metaclass(abc.ABCMeta, FiniteOutcomeModel)):
    r"""
    A system whose model parameters and experiment parameters together 
    describe the generator of dynamics of a quantum system. 

    :param preparations: A list of possible state preparations, each of 
        type ``qutip.Qobj``.
    :param measurements: A list of measurements that can be performed. Each 
        measurement should be a list of ``qutip.Qobj`` 
        which form a POVM, or a single ``qutip.Qobj`` 
        specifying one of the two POVM elements in a two-outcome 
        measurement.
    """

    def __init__(self, preparations, measurements):
        super(QHLModel, self).__init__()

        self._preparations = preparations
        self._measurements = measurements

        def povm_count(povm):
            try:
                return len(povm)
            except TypeError:
                return 2

        self._povm_counts = np.array([
            povm_count(self.measurements[idx_meas])
            for idx_meas in range(len(self.measurements))
        ])

        self._two_outcome_measurement = self._povm_counts == 2
        self._is_n_outcomes_constant = np.all(self._povm_counts = self._povm_counts[0])

        self._domains = [
            IntegerDomain(min=0, max=x-1)
            for x in self._povm_counts
        ]

    ## ABSTRACT PROPERTIES ##

    @abc.abstractproperty
    def system_expparams_dtype(self):
        """
        Returns the dtype of the experiment parameter array that 
        does not have to do with which preparation or measurement 
        we are performing, for example [('tp','float64')] for 
        a Rabi experiment.
        """
        pass


    ## CONCRETE PROPERTIES ##

    @property
    def expparams_dtype(self):
        return self.system_expparams_dtype + [('idx_prep',int),('idx_meas',int)]

    @property
    def n_measurements(self):
        """
        Number of possible measurement sets.
        """
        return len(self.measurements)

    @property
    def n_preparations(self):
        """
        Number of possible state preparations.
        """
        return len(self.preparations)

    @property
    def measurements(self):
        """
        List of measurements that can be performed. Each 
        measurement should be a list of ``qutip.Qobj`` 
        which form a POVM, or a single ``qutip.Qobj`` 
        specifying one of the two POVM elements in a two-outcome 
        measurement.
        """
        return self._measurements

    @property
    def preparations(self):
        """
        List of ``qutip.Qobj`` objects specifying possible 
        initial quantum states, pure or mixed.
        """
        return self._preparations

    @property
    def is_n_outcomes_constant(self):
        return self._is_n_outcomes_constant

    
    ## ABSTRACT METHODS ##    

   
    ## CONCRETE METHODS ##

    def n_outcomes(self, expparams):
        return self._povm_counts[expparams['idx_meas']]

    def domain(self, expparams):
        return [self._domains[idx_meas] for idx_meas in expparams['idx_meas']]

    def are_expparams_dtypes_consistent(self, expparams):
        # outcome types consistent iff the number of outcomes is the same.
        return self.is_n_outcomes_constant


class SimpleQHLModel(QHLModel):

    def __init__(self, observables, preparations, solver):
        super(SimpleQHLModel, self).__init__(
                observables=observables, 
                preparations=preparations
            )
    
    @abc.abstractproperty
    def system_expparams_dtype(self):
        """
        Returns the dtype of the experiment parameter array that 
        does not have to do with which preparation or measurement 
        we are performing, for example [('tp','float64')] for 
        a Rabi experiment.
        """
        return [('t','float64')]

    @abc.abstractmethod
    def hamiltonian(self, modelparams, expparams):
        """
        Returns a hamiltonian for the given modelparam and expparam
        """
        pass

    def solver_args(self, modelparams, expparams):
