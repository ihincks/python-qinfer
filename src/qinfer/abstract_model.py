#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# abstract_model.py: Abstract interfaces for models with different levels of
#     functionality.
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

## FEATURES ##################################################################

from __future__ import absolute_import
from __future__ import division, unicode_literals

## EXPORTS ###################################################################

__all__ = [
    'Model',
    'FiniteOutcomeModel',
    'DifferentiableModel'
]

## IMPORTS ###################################################################

from builtins import range, map
from future.utils import with_metaclass

import abc
    # Python standard library package for specifying abstract classes.
import numpy as np

from qinfer.utils import safe_shape
    
## CLASSES ###################################################################

class Model(with_metaclass(abc.ABCMeta, object)):
    """
    Represents a system which can be simulated according to
    various model parameters and experimental control parameters
    in order to produce representative data.

    See :ref:`models_guide` for more details.

    :param bool always_resample_outcomes: Resample outcomes stochastically with 
            each outcome call.

    :param :class:`~numpy.ndarray` initial_outcomes: Initial set of outcomes 
            that may be supplied. Otherwise initial outcomes default to 
            zeros. 
    """
    def __init__(self,always_resample_outcomes=False,initial_outcomes = None):
        """
        Initialize Model model
        :param bool always_resample_outcomes: Resample outcomes stochastically with 
                    each outcome call.
        :param :class:`~numpy.ndarray` initial_outcomes: Initial set of outcomes 
                    that may be supplied. Otherwise initial outcomes default to 
                    zeros. 
        """
        self._sim_count = 0
        self._call_count = 0
        if initial_outcomes is not None:
            #verify that the supplied outcomes are of the correct length
            assert initial_outcomes.dtype== self.outcomes_dtype
            self._outcomes = initial_outcomes
        else:
            #otherwise initialize all outcomes to zero at start
            self._outcomes = np.empty(None,self.outcomes_dtype)

        self.needs_resample = False
        # Initialize a default scale matrix.
        self._Q = np.ones((self.n_modelparams,))
        
    ## ABSTRACT PROPERTIES ##
    
    @abc.abstractproperty
    def n_modelparams(self):
        """
        Returns the number of real model parameters admitted by this model.
        
        This property is assumed by inference engines to be constant for
        the lifetime of a :class:`Model` instance.
        """
        pass
        
    @abc.abstractproperty
    def expparams_dtype(self):
        """
        Returns the dtype of an experiment parameter array. For a
        model with single-parameter control, this will likely be a scalar dtype,
        such as ``"float64"``. More generally, this can be an example of a
        record type, such as ``[('time', 'float64'), ('axis', 'uint8')]``.
        
        This property is assumed by inference engines to be constant for
        the lifetime of a Model instance.
        """
        pass

            
    ## CONCRETE PROPERTIES ##

    @property
    def outcomes_dtype(self):
        """
        Returns the dtype of the outcomes parameter array. For a
        model with single-parameter outcomes, this will likely be a scalar dtype,
        such as ``"int64"`` for finite models or ``"float64"`` for continuous models. 
        More generally, this can be an example of a record type, such as 
        ``[('outcome x', 'uint8'), ('outcome y', 'uint8')]``.
        
        This property is assumed by inference engines to be constant for
        the lifetime of a Model instance.
        """
        return 'uint32'
    
    @property
    def is_n_outcomes_constant(self):
        """
        Returns ``True`` if and only if the number of outcomes for each
        experiment is independent of the experiment being performed.
        
        This property is assumed by inference engines to be constant for
        the lifetime of a Model instance.
        """
        return True

    @property
    def model_chain(self):
        """
        Returns a tuple of models upon which this model is based,
        such that properties and methods of underlying models for
        models that decorate other models can be accessed. For a
        standalone model, this is always the empty tuple.
        """
        return ()

    @property
    def base_model(self):
        """
        Returns the most basic model that this model depends on.
        For standalone models, this property satisfies ``model.base_model is model``.
        """
        return self

    @property
    def underlying_model(self):
        """
        Returns the model that this model is based on (decorates)
        if such a model exists, or ``None`` if this model is
        independent.
        """
        return self.model_chain[-1] if self.model_chain else None
    
    @property
    def sim_count(self):
        """
        Returns the number of data samples that have been produced by
        this simulator.

        :rtype: int
        """
        return self._sim_count
        
    @property
    def Q(self):
        r"""
        Returns the diagonal of the scale matrix :math:`\matr{Q}` that
        relates the scales of each of the model parameters. In particular,
        the quadratic loss for this Model is defined as:
        
        .. math::
            L_{\matr{Q}}(\vec{x}, \hat{\vec{x}}) =
            (\vec{x} - \hat{\vec{x}})^\T \matr{Q} (\vec{x} - \hat{\vec{x}})

        If a subclass does not explicitly define the scale matrix, it is taken
        to be the identity matrix of appropriate dimension.
        
        :return: The diagonal elements of :math:`\matr{Q}`.
        :rtype: :class:`~numpy.ndarray` of shape ``(n_modelparams, )``.
        """
        return self._Q
        
    @property
    def modelparam_names(self):
        """
        Returns the names of the various model parameters admitted by this
        model, formatted as LaTeX strings.
        """
        return list(map("x_{{{}}}".format, range(self.n_modelparams)))

    @property
    def needs_outcome_resample(self):
        """
        Determines whether the outcomes needs to be resampled during call 
        to :func:`~abstract_model.Model.outcomes`
        .
        :return: Resampling state 
        :rtype: bool 
        """
        return self._needs_outcome_resample

    @needs_outcome_resample.setter
    def needs_outcome_resample(self, needs_outcome_resample):
        """
        Set outcome resampling flag.

        :param bool needs_outcome_resample: Whether to resample or not in call 
        to :func:`~abstract_model.Model.outcomes`.
        """
        self._needs_outcome_resample = needs_outcome_resample

    @property
    def call_count(self):
        # TODO: document
        return self._call_count
    
    ## CONCRETE METHODS ##

    def _repr_html_(self, suppress_base=False):
        s = r"""
            <strong>{type.__name__}</strong> at 0x{id:0x}: {n_mp} model parameter{plural}
        """.format(
            id=id(self), type=type(self),
            n_mp=self.n_modelparams,
            plural="" if self.n_modelparams == 1 else "s"
        )
        if not suppress_base and self.model_chain:
            s += r"""<br>
            <p>Model chain:</p>
            <ul>{}
            </ul>
            """.format(r"\n".join(
                u"<li>{}</li>".format(model._repr_html_(suppress_base=True))
                for model in reversed(self.model_chain)
            ))
        return s
    
    ## ABSTRACT METHODS ##
    
    @abc.abstractmethod
    def n_outcomes(self, expparams=None):
        """
        Returns an array of dtype ``uint`` describing the number of outcomes
        for each experiment specified by ``expparams``.
        If there are an infinite (or intractibly large) number of outcomes, 
        this value specifies the number of outcomes to randomly sample
        
        :param numpy.ndarray expparams: Array of experimental parameters. This
            array must be of dtype agreeing with the ``expparams_dtype``
            property.
        """
        pass
    
    @abc.abstractmethod
    def are_models_valid(self, modelparams):
        """
        Given a shape ``(n_models, n_modelparams)`` array of model parameters,
        returns a boolean array of shape ``(n_models)`` specifying whether
        each set of model parameters represents is valid under this model.
        """
        pass
        
    @abc.abstractmethod
    def simulate_experiment(self, modelparams, expparams, repeat=1):
        """
        Produces data according to the given model parameters and experimental
        parameters, structured as a NumPy array.

        :param np.ndarray modelparams: A shape ``(n_models, n_modelparams)``
            array of model parameter vectors describing the hypotheses under
            which data should be simulated.
        :param np.ndarray expparams: A shape ``(n_experiments, )`` array of
            experimental control settings, with ``dtype`` given by 
            :attr:`~qinfer.Model.expparams_dtype`, describing the
            experiments whose outcomes should be simulated.
        :param int repeat: How many times the specified experiment should
            be repeated.
        :rtype: np.ndarray
        :return: A three-index tensor ``data[i, j, k]``, where ``i`` is the repetition,
            ``j`` indexes which vector of model parameters was used, and where
            ``k`` indexes which experimental parameters where used. If ``repeat == 1``,
            ``len(modelparams) == 1`` and ``len(expparams) == 1``, then a scalar
            datum is returned instead.
        """
        self._sim_count += modelparams.shape[0] * expparams.shape[0] * repeat
    
    @abc.abstractmethod
    def likelihood(self, outcomes, modelparams, expparams):
        r"""
        Calculates the probability of each given outcome, conditioned on each
        given model parameter vector and each given experimental control setting.
        :param np.ndarray modelparams: A shape ``(n_models, n_modelparams)``
            array of model parameter vectors describing the hypotheses for
            which the likelihood function is to be calculated.
        :param np.ndarray expparams: A shape ``(n_experiments, )`` array of
            experimental control settings, with ``dtype`` given by 
            :attr:`~qinfer.Model.expparams_dtype`, describing the
            experiments from which the given outcomes were drawn.
        :rtype: np.ndarray
        :return: A three-index tensor ``L[i, j, k]``, where ``i`` is the outcome
            being considered, ``j`` indexes which vector of model parameters was used,
            and where ``k`` indexes which experimental parameters where used.
            Each element ``L[i, j, k]`` then corresponds to the likelihood
            :math:`\Pr(d_i | \vec{x}_j; e_k)`.
        """
        
        # Count the number of times the inner-most loop is called.
        self._call_count += (
            safe_shape(outcomes) * safe_shape(modelparams) * safe_shape(expparams)
        )

    ## CONCRETE METHODS ##
    
    def clear_cache(self):
        """
        Tells the model to clear any internal caches used in computing
        likelihoods and drawing samples. Calling this method should not cause
        any different results, but should only affect performance.
        """
        # By default, no cache to clear.
        pass
    
    def experiment_cost(self, expparams):
        """
        Given an array of experimental parameters, returns the cost associated
        with performing each experiment. By default, this cost is constant
        (one) for every experiment.
        
        :param expparams: An array of experimental parameters for which the cost
            is to be evaluated.
        :type expparams: :class:`~numpy.ndarray` of ``dtype`` given by
            :attr:`~Model.expparams_dtype`
        :return: An array of costs corresponding to the specified experiments.
        :rtype: :class:`~numpy.ndarray` of ``dtype`` ``float`` and of the
            same shape as ``expparams``.
        """
        return np.ones(expparams.shape)
        
    def distance(self, a, b):
        r"""
        Gives the distance between two model parameter vectors :math:`\vec{a}` and
        :math:`\vec{b}`. By default, this is the vector 1-norm of the difference
        :math:`\mathbf{Q} (\vec{a} - \vec{b})` rescaled by
        :attr:`~Model.Q`.
        
        :param np.ndarray a: Array of model parameter vectors having shape
            ``(n_models, n_modelparams)``.
        :param np.ndarray b: Array of model parameters to compare to, having
            the same shape as ``a``.
        :return: An array ``d`` of distances ``d[i]`` between ``a[i, :]`` and
            ``b[i, :]``.
        """
        
        return np.apply_along_axis(
            lambda vec: np.linalg.norm(vec, 1),
            1,
            self.Q * (a - b)
        )
        
    def update_timestep(self, modelparams, expparams):
        r"""
        Returns a set of model parameter vectors that is the update of an
        input set of model parameter vectors, such that the new models are
        conditioned on a particular experiment having been performed.
        By default, this is the trivial function
        :math:`\vec{x}(t_{k+1}) = \vec{x}(t_k)`.
        
        :param np.ndarray modelparams: Set of model parameter vectors to be
            updated.
        :param np.ndarray expparams: An experiment parameter array describing
            the experiment that was just performed.
        
        :return np.ndarray: Array of shape
            ``(n_models, n_modelparams, n_experiments)`` describing the update
            of each model according to each experiment.
        """
        return modelparams[:, :, np.newaxis]

    def canonicalize(self, modelparams):
        r"""
        Returns a canonical set of model parameters corresponding to a given
        possibly non-canonical set. This is used for models in which there
        exist model parameters :math:`\vec{x}_i` and :\math:`\vec{x}_j` such
        that

        .. math::

            \Pr(d | \vec{x}_i; \vec{e}) = \Pr(d | \vec{x}_j; \vec{e})

        for all outcomes :math:`d` and experiments :math:`\vec{e}`. For
        models admitting such an ambiguity, this
        method should then be overridden to return a consistent choice
        out of such vectors, hence avoiding supurious model degeneracies.

        Note that, by default, :class:`~qinfer.smc.SMCUpdater` will *not*
        call this method.
        """
        return modelparams

    def _resample_outcomes(self, weights, modelparams, expparams):
        """
        Randomly sample modelparams according to the given weights, and then use these 
        sampled points to sample outcomes from the likelihood function. Ie. sample 
        :math:`n` points from 
        .. :math::
            \vec{x_i} ~ \pi(\vec{x})
             y_i ~ L(\vec{x_i};\vec{C}) 

        where :math:`\vec{x_i}` is a sampled point from the given particle distribution, 
        and :math:`y_i` is the sampled outcome from this point. In the limit of 
        infinite samples the binned outcomes should be proportional to 
        the outcome likelihood distribution under the prior probability function. 


        :param np.ndarray weights: Set of weights with a weight
            corresponding to every modelparam. 
        :param np.ndarray modelparams: Set of model parameter vectors (particles) 
            to samples outcomes from.
        :param np.ndarray expparams: An experiment parameter array describing
            the experiments that outcomes should be sampled for.
        """

        sampled_points = modelparams[np.random.choice(np.shape(modelparams)[0],size=self.n_outcomes,p=weights)]
        outcomes = self.simulate_experiment(sample_points,expparams)

        assert outcomes.dtype == self.outcomes_dtype
        self._outcomes =  outcomes.reshape(sampled_points.shape[0],expparams.shape[0])

    def outcomes(self, weights, modelparams, expparams, resample=False):
        """
        Randomly sample modelparams according to the given weights, and then use these 
        sampled points to sample outcomes from the likelihood function.
        Only resamples if ``needs_outcome_resample``,``resample``, or ``self.always_resample_outcomes`` 
        is ``True``, otherwise returns the cached value.

        :param np.ndarray weights: Set of weights with a weight
            corresponding to every modelparam. 
        :param np.ndarray modelparams: Set of model parameter vectors (particles) to samples outcomes from.
        :param np.ndarray expparams: An experiment parameter array describing
            the experiments that outcomes should be sampled for.
        :param bool resample: Force resampling of outcomes 
        :return np.ndarray: Array of shape ``(n_outcomes,n_expparams)`` listing the 
        sampled outcomes, of type ``outcomes_dtype``.
        """
        if self.needs_outcome_resample or resample or self.always_resample_outcomes:
            self._resample_outcomes(weights,modelparams,expparams)

        return self._outcomes

        
        
class LinearCostModelMixin(Model):
    # FIXME: move this mixin to a new module.
    # TODO: test this mixin.
    """
    This mixin implements :meth:`Model.experiment_cost` by setting the
    cost of an experiment equal to the value of a given field of each
    ``expparams`` element (by default, ``t``).
    """
    _field = "t"
    
    def experiment_cost(self, expparams):
        return expparams[self._field]

class FiniteOutcomeModel(Model):
    """
    Represents a system which can be simulated according to
    various model parameters and experimental control parameters
    in order to produce the probability of a hypothetical data
    record. As opposed to :class:`~qinfer.Simulatable`, instances
    of :class:`~qinfer.Model` not only produce data consistent
    with the description of a system, but also evaluate the probability
    of that data arising from the system.

    See :ref:`models_guide` for more details.
    """
    
    ## INITIALIZERS ##
    def __init__(self):
        super(FiniteOutcomeModel, self).__init__()
    
    ## CONCRETE PROPERTIES ##

    @property
    def call_count(self):
        """
        Returns the number of points at which
        the probability of this model has been evaluated,
        where a point consists of a hypothesis about the model (a vector of
        model parameters), an experimental control setting (expparams) and
        a hypothetical or actual datum.

        :rtype: int
        """
        return self._call_count
    
    ## ABSTRACT METHODS ##
                
    ## CONCRETE METHODS ##
    # These methods depend on the abstract methods, and thus their behaviors
    # change in each inheriting class.
    
    def is_model_valid(self, modelparams):
        """
        Returns True if and only if the model parameters given are valid for
        this model.
        """
        return self.are_models_valid(modelparams[np.newaxis, :])[0]

    def simulate_experiment(self, modelparams, expparams, repeat=1):
        # NOTE: implements abstract method of Model.
        # TODO: document
        
        # Call the superclass simulate_experiment, not recording the result.
        # This is used to count simulation calls.
        super(FiniteOutcomeModel, self).simulate_experiment(modelparams, expparams, repeat)
        
        if self.is_n_outcomes_constant:
            all_outcomes = np.arange(self.n_outcomes(expparams[0, np.newaxis]))
            probabilities = self.likelihood(np.arange(self.n_outcomes(expparams)), modelparams, expparams)
            cdf = np.cumsum(probabilities,axis=0)
            randnum = np.random.random((repeat, 1, modelparams.shape[0], expparams.shape[0]))
            outcomes = np.argmax(cdf > randnum, axis=1)
        else:
            # Loop over each experiment, sadly.
            outcomes = np.empty((repeat, modelparams.shape[0], expparams.shape[0]))
            for idx_experiment, single_expparams in enumerate(expparams[:, np.newaxis]):
                all_outcomes = np.arange(self.n_outcomes(single_expparams))
                
                probabilities = self.likelihood(np.arange(self.n_outcomes(single_expparams)), modelparams, single_expparams)
                cdf = np.cumsum(probabilities, axis=0)[..., 0]
                randnum = np.random.random((repeat, 1, modelparams.shape[0]))
                outcomes[:, :, idx_experiment] = np.argmax(cdf > randnum, axis=1)
                
        return outcomes[0, 0, 0] if repeat == 1 and expparams.shape[0] == 1 and modelparams.shape[0] == 1 else outcomes
                
    ## STATIC METHODS ##
    # These methods are provided as a convienence to make it easier to write
    # simple models.
    
    @staticmethod
    def pr0_to_likelihood_array(outcomes, pr0):
        """
        Assuming a two-outcome measurement with probabilities given by the
        array ``pr0``, returns an array of the form expected to be returned by
        ``likelihood`` method.
        
        :param numpy.ndarray outcomes: Array of integers indexing outcomes.
        :param numpy.ndarray pr0: Array of shape ``(n_models, n_experiments)``
            describing the probability of obtaining outcome ``0`` from each
            set of model parameters and experiment parameters.
        """
        pr0 = pr0[np.newaxis, ...]
        pr1 = 1 - pr0

        if len(np.shape(outcomes)) == 0:
            outcomes = np.array(outcomes)[None]
                    
        return np.concatenate([
            pr0 if outcomes[idx] == 0 else pr1
            for idx in range(safe_shape(outcomes))
        ]) 
        
class DifferentiableModel(with_metaclass(abc.ABCMeta, Model)):
    
    @abc.abstractmethod
    def score(self, outcomes, modelparams, expparams, return_L=False):
        r"""
        Returns the score of this likelihood function, defined as:
        
        .. math::
        
            q(d, \vec{x}; \vec{e}) = \vec{\nabla}_{\vec{x}} \log \Pr(d | \vec{x}; \vec{e}).
            
        Calls are represented as a four-index tensor
        ``score[idx_modelparam, idx_outcome, idx_model, idx_experiment]``.
        The left-most index may be suppressed for single-parameter models.
        
        If return_L is True, both `q` and the likelihood `L` are returned as `q, L`.
        """
        pass
        
    def fisher_information(self, modelparams, expparams):
        """
        Returns the covariance of the score taken over possible outcomes,
        known as the Fisher information.
        
        The result is represented as the four-index tensor
        ``fisher[idx_modelparam_i, idx_modelparam_j, idx_model, idx_experiment]``,
        which gives the Fisher information matrix for each model vector
        and each experiment vector.
        
        .. note::
            
            The default implementation of this method calls
            :meth:`~DifferentiableModel.score()` for each possible outcome,
            which can be quite slow. If possible, overriding this method can
            give significant speed advantages.
        """
        
        # TODO: break into two cases, one for constant outcomes, one for
        #       variable. The latter will have to be a loop, which is much
        #       slower.
        #       Here, we sketch the first case.
        # FIXME: completely untested!
        if self.is_n_outcomes_constant:
            outcomes = np.arange(self.n_outcomes(expparams))
            scores, L = self.score(outcomes, modelparams, expparams, return_L=True)
            
            assert len(scores.shape) in (3, 4)
            
            if len(scores.shape) == 3:
                scores = scores[np.newaxis, :, :, :]
            
            # Note that E[score] = 0 by regularity assumptions, so we only
            # need the expectation over the outer product.
            return np.einsum("ome,iome,jome->ijme",
                L, scores, scores
            )
        else:
            # Indexing will be a major pain here, so we need to start
            # by making an empty array, so that index errors will be raised
            # when (not if!) we make mistakes.
            fisher = np.empty((
                self.n_modelparams, self.n_modelparams,
                modelparams.shape[0], expparams.shape[0]
            ))
            
            # Now we loop over experiments, since we cannot vectorize the
            # expectation value over data.
            for idx_experiment, experiment in enumerate(expparams):
                experiment = experiment.reshape((1,))
                n_o = self.n_outcomes(experiment)
            
                outcomes = np.arange(n_o)
                scores, L = self.score(outcomes, modelparams, experiment, return_L=True)
                
                fisher[:, :, :, idx_experiment] = np.einsum("ome,iome,jome->ijme",
                    L, scores, scores
                )
            
            return fisher
            
            
