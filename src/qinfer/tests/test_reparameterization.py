#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# test_reparameterization.py: Checks that ModelReparameterization works 
#   as expected.
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

## IMPORTS ####################################################################

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal, assert_array_less

from qinfer.tests.base_test import (
    DerandomizedTestCase
)
import abc
from qinfer import (
    ModelReparameterization, BoundedReparameterization
)

import unittest

## CONSTANTS ###################################################################

## DOMAIN TESTS ################################################################

class TestBoundedReparameterization(DerandomizedTestCase):
    """
    Tests the BoundedReparameterization subclass of ModelReparameterization.
    """
    
    def test_conversion(self):
        
        bounds = [
            [-np.inf,np.inf],
            [0,1], [-3,12],
            [-np.inf,5],
            [2,np.inf],
            [-np.inf,np.inf]
        ]
        br = BoundedReparameterization(bounds)
        
        unnatural_mps = np.random.uniform(low=-5, high=5, size=(10000, len(bounds)))
        natural_mps = br.to_natural(unnatural_mps)
        #import pdb; pdb.set_trace()
        
        assert_almost_equal(br.from_natural(natural_mps), unnatural_mps)
        
        for idx, bound in enumerate(bounds):
            assert_array_less(natural_mps,100000)
            assert_array_less(bound[0] - 1e-6, natural_mps)
        
        
        
