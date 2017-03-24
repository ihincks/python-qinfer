#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# reparameterizations.py: module for reparameterizations of
#   of models.
##
# © 2012 Chris Ferrie (csferrie@gmail.com) and
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

## IMPORTS ###################################################################

from __future__ import division
from __future__ import absolute_import

from builtins import range
from future.utils import with_metaclass

import abc
import numpy as np
from scipy.special import logit, expit

import warnings

## EXPORTS ###################################################################

__all__ = [
    'ModelReparameterization',
    'BoundedReparameterization'
]

## FUNCTIONS #################################################################

def apply_1d_function_list_to_array(fcns, array, out=None):
    if len(fcns) != array.shape[-1]:
        raise ValueError('Function list length ({}) and array size ({}) incompatible'.format(len(fcns), array.shape))
    out = np.empty(array.shape) if out is None else out
    for idx, fcn in enumerate(fcns):
        out[..., idx] = fcn(array[..., idx])
    return out
    
def to_compact(lower=0, upper=1, mean=None, std=None):
    # returns a function taking real numbers to [lower,upper]
    if mean is None:
        return lambda y, lower=lower, diff=upper-lower: \
            lower + diff * expit(y)
    else:
        std = 1 if std is None else std
        d = std * (1 / (upper - mean) + 1 / (mean - lower))
        mt = logit((mean - lower) / (upper - lower))
        return lambda y, lower=lower, diff=upper-lower, d=d, mt=mt: \
            lower + diff * expit(d * y + mt)
def from_compact(lower=0, upper=1, mean=None, std=None): 
    # returns a function taking [lower, upper] to all real numbers
    if mean is None and std is None:
        return lambda x, lower=lower, diff=upper-lower: \
            logit((x - lower) / diff)
    else:
        std = 1 if std is None else std
        d = std * (1 / (upper - mean) + 1 / (mean - lower))
        mt = logit((mean - lower) / (upper - lower))
        return lambda x, lower=lower, diff=upper-lower, d=d, mt=mt: \
            (logit((x - lower) / diff) - mt) / d
            
def to_lbounded(lower=0, mean=None, std=None):
    # returns a function taking real numbers to [lower,inf]
    if mean is None:
        return lambda y, lower=lower: lower + np.exp(y)
    else:
        std = 1 if std is None else std
        d = std * 1 / (mean - lower)
        mt = np.log(mu - lower)
        return lambda y, lower=lower, d=d, mt=mt: lower + np.exp(d * y + mt)
def from_lbounded(lower=0, mean=None, std=None): 
    # returns a function taking [lower, inf] to all real numbers
    if mean is None and std is None:
        return lambda x, lower=lower: np.log(x - lower)
    else:
        std = 1 if std is None else std
        d = std * 1 / (mean - lower)
        mt = np.log(mu - lower)
        return lambda x, lower=lower, d=d, mt=mt: (np.log(x - lower) - mt) / d
            
def to_ubounded(upper=0, mean=None, std=None):
    # returns a function taking real numbers to [-inf,upper]
    if mean is None:
        return lambda y, upper=upper: upper - np.exp(y)
    else:
        std = 1 if std is None else std
        d = std * 1 / (mean - upper)
        mt = np.log(upper - mu)
        return lambda y, upper=upper, d=d, mt=mt: upper - np.exp(d * y + mt)
def from_ubounded(upper=0, mean=None, std=None): 
    # returns a function taking [-inf, upper] to all real numbers
    if mean is None and std is None:
        return lambda x, upper=upper: np.log(upper - x)
    else:
        std = 1 if std is None else std
        d = std * 1 / (mean - upper)
        mt = np.log(upper - mu)
        return lambda x, upper=upper, d=d, mt=mt: (np.log(upper - x) - mt) / d

## ABSTRACT CLASSES AND MIXINS ###############################################

class ModelReparameterization(object):
    r"""
    The paradigm of reparameterizations is that the model has
    model parameters which are naturally parameterized (due to physics,
    probability theory, etc), but for which there exists a computationally nicer 
    parameterization (less correlation between parameters, more 
    gaussian in shape, no hard cutoffs to the region, etc).
    
    This class provides a reparameterization function, its 
    inverse, and its jacobian, which is needed to modify the 
    likelihood.
    
    :param bool do_hash: If ``True``, memoization of this class' functions
        is performed for a history length of 1. Model parameters are hashed with
        ``hash(str(modelparams))``, and if the previous evalution had the 
        same hash, the same result is returned. It is important to turn this
        value to ``False`` if you expect hash collisions (or override the 
        ``_hash`` function).
    """
    def __init__(self, do_hash=True):
        self._do_hash = do_hash
        self._to_natural_hash = 0
        self._from_natural_hash = 0
        self._to_natural_jac_hash = 0
        self._to_natural_memo = None
        self._from_natural_memo = None
        self._to_natural_jac_memo = None
    
    def _hash(self, modelparams):
        return hash(str(modelparams))
        
    @abc.abstractmethod
    def _to_natural(self, modelparams):
        """
        Converts the given model parameters from the unnatural parameterization
        to the natural parameterization.
        
        :param np.ndarray modelparams: A shape ``(n_models, n_modelparams)``
            array of model parameters in the unnatural parameterization.
            
        :rtype: np.ndarray
        :return: An array of the same shape as the input but in the 
             natural parameterization.
        """
        pass
            
        
    @abc.abstractmethod    
    def _from_natural(self, modelparams):
        """
        Converts the given model parameters from the natural parameterization
        to the unnatural parameterization.
        
        :param np.ndarray modelparams: A shape ``(n_models, n_modelparams)``
            array of model parameters in the natural parameterization.
            
        :rtype: np.ndarray
        :return: An array of the same shape as the input but in the 
             unnatural parameterization.
        """
        pass
    
    @abc.abstractmethod    
    def _to_natural_jac(self, modelparams):
        """
        The jacobian of ``to_natural`` at the given model parameters.
        
        :param np.ndarray modelparams: A shape ``(n_models, n_modelparams)``
            array of model parameters in the unnatural parameterization.
            
        :rtype: np.ndarray
        :return: A shape ``(n_models, n_modelparams, n_modelparams)``
            array.
        """
        pass
        
    def to_natural(self, modelparams):
        """
        Converts the given model parameters from the unnatural parameterization
        to the natural parameterization.
        
        :param np.ndarray modelparams: A shape ``(n_models, n_modelparams)``
            array of model parameters in the unnatural parameterization.
            
        :rtype: np.ndarray
        :return: An array of the same shape as the input but in the 
             natural parameterization.
        """
        if self._do_hash:
            new_hash = self._hash(modelparams)
            if new_hash != self._to_natural_hash:
                self._to_natural_hash = new_hash
                self._to_natural_memo = self._to_natural(modelparams)
            return self._to_natural_memo
        else:
            return self._to_natural(modelparams)
              
    def from_natural(self, modelparams):
        """
        Converts the given model parameters from the natural parameterization
        to the unnatural parameterization.
        
        :param np.ndarray modelparams: A shape ``(n_models, n_modelparams)``
            array of model parameters in the natural parameterization.
            
        :rtype: np.ndarray
        :return: An array of the same shape as the input but in the 
             unnatural parameterization.
        """
        if self._do_hash:
            new_hash = self._hash(modelparams)
            if new_hash != self._from_natural_hash:
                self._from_natural_hash = new_hash
                self._from_natural_memo = self._from_natural(modelparams)
            return self._from_natural_memo
        else:
            return self._from_natural(modelparams)  
      
    def to_natural_jac(self, modelparams):
        """
        The jacobian of ``to_natural`` at the given model parameters.
        
        :param np.ndarray modelparams: A shape ``(n_models, n_modelparams)``
            array of model parameters in the unnatural parameterization.
            
        :rtype: np.ndarray
        :return: A shape ``(n_models, n_modelparams, n_modelparams)``
            array.
        """
        if self._do_hash:
            new_hash = self._hash(modelparams)
            if new_hash != self._to_natural_jac_hash:
                self._to_natural_jac_hash = new_hash
                self._to_natural_jac_memo = self._to_natural_jac(modelparams)
            return self._to_natural_jac_memo
        else:
            return self._to_natural_jac(modelparams)

## CLASSES ###################################################################

class BoundedReparameterization(ModelReparameterization):
    r"""
    A reparameterization where some of the model parameters have 
    hard cutoffs on one or both sides. Parameters with cutoffs
    on both sides are reparameterized with sigmoid logistic
    functions, and parameters bounded only on one side 
    are reparameterized with 
    
    :param np.ndarray bounds: An array ``[bounds1, bounds2,..]``
        where each of the bounds is of the form ``[min, max]``, 
        ``[min, max, mean]``, or ``[min, max, mean, std]``,
        with one entry for every model parameter. Use 
        ``np.inf`` and ``-np.inf`` for one-sided or no-sided 
        values of ``min`` and ``max``. If provided, the reparameterization is 
        shifted by ``mean`` and scaled by ``std``.
    """
    
    def __init__(self, bounds, do_hash=True):
        super(BoundedReparameterization, self).__init__(do_hash)
        self._bounds = bounds
        
        # now we go through every parameter and make a 1D transformation
        self._to_natural_list = []
        self._from_natural_list = []
        for bound in bounds:
            if bound is None:
                lower, upper, mean, std = -np.inf, np.inf, np.inf, np.inf
            elif len(bound) == 2:
                lower, upper = bound
                mean = (upper + lower) / 2
                std = upper - lower
            elif len(bound) == 3:
                lower, upper, mean = bound
                std = upper - lower
            elif len(bound) == 4:
                lower, upper, mean, std = bound
                
            std = std if np.isfinite(std) else None
            mean = mean if np.isfinite(mean) else None
            
            if np.isinf(lower) and np.isinf(upper):
                # not constrained
                if mean is None and std is None:
                    self._to_natural_list.append(lambda y: y)
                    self._from_natural_list.append(lambda x: x)
                else:
                    self._to_natural_list.append(lambda y, mean=mean, std=std: y * std + mean)
                    self._from_natural_list.append(lambda x, mean=mean, std=std: (x - mean) / std)
            elif np.isinf(lower):
                # bounded above
                self._to_natural_list.append(to_ubounded(upper=upper, mean=mean, std=std))
                self._from_natural_list.append(from_ubounded(upper=upper, mean=mean, std=std))                  
            elif np.isinf(upper):
                # bounded below
                self._to_natural_list.append(to_lbounded(lower=lower, mean=mean, std=std))
                self._from_natural_list.append(from_lbounded(lower=lower, mean=mean, std=std))
            else:
                # bounded on both sides
                self._to_natural_list.append(to_compact(lower=lower, upper=upper, mean=mean, std=std))
                self._from_natural_list.append(from_compact(lower=lower, upper=upper, mean=mean, std=std))
    
    
    def _to_natural(self, modelparams):
        return apply_1d_function_list_to_array(self._to_natural_list, modelparams)
    def _from_natural(self, modelparams):
        return apply_1d_function_list_to_array(self._from_natural_list, modelparams)
    def _to_natural_jac(self, modelparams):
        return apply_1d_function_list_to_array(self._to_natural_jac_list, modelparams)
