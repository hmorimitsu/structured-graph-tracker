###############################################################################
#
# Copyright (c) 2016, Henrique Morimitsu,
# University of Sao Paulo, Sao Paulo, Brazil
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# #############################################################################

import cv2
import math
import numpy as np
from ParticleFilter import ParticleFilter
from ExtendedParticleFilter import ExtendedParticleFilter
from ObjectTracker import ObjectTracker
from Rectangle import Rectangle


class PFTracker(ObjectTracker):
    """ This class implements a visual tracker based on color histogram with
    particle filter. """
    def __init__(self, initial_bb, num_particles, num_states, dynamics_matrix,
                 particle_lower_bounds, particle_upper_bounds,
                 noise_type='gaussian', noise_param1=None, noise_param2=None,
                 maximum_total_weight=0.0,
                 final_state_decision_method='weighted_average',
                 noise_dispersion_based_on_weight=False, dispersion_factor=5.0,
                 minimum_dispersion=0.5):
        """ dynamics_matrix is a ns x ns square matrix, where ns = num_states
        particle_lower_bounds is a vector that represents
            the minimum values of each state
        particle_upper_bounds is a vector that represents
            the maximum values of each state
        noise_type must be either 'gaussian' or 'uniform'
        noise_param1 must be either None of a vector with num_states elements.
            If it is set as None, then it is initialized as a vector of zeros.
            If noise_type is gaussian, this parameter represents the means of
            the noise distribution, while if the noise_type is uniform, then
            it represents the lower bounds of the interval
        noise_param2 is similar to noise_param1. If it is None, it is set as a
            vector of ones. When the noise_type is gaussian, it represents the
            standard deviations, while if it is uniform, it is the upper bounds
            of the interval
        maximum_total_weight is the highest value that it is expected to be
            obtained by summing all the weights. This parameter is necessary if
            the weighting function is not bounded or if the real maximum value
            is never reached in real situations. This value should be set as the
            highest value that usually occurs
        final_state_decision_method must be either 'best', 'average' or
            'weighted_average'. If best, the particle with highest weight is
            chosen as the new state. If average, the new state is computed
            from the simple average of all the particles. If weighted_average,
            the state comes from an average of all particles averaged by
            their weights
        noise_dispersion_based_on_weight if set as False, then this class
            behaves the same way as the ParticleFilter parent class
        dispersion_factor adjusts the variance of the noise, the higher the
            dispersion_factor, the higher the variance.
        minimum_dispersion is the lower bound of the noise dispersion
        """
        self._pf = ExtendedParticleFilter(
            num_particles, num_states, dynamics_matrix, particle_lower_bounds,
            particle_upper_bounds, noise_type, noise_param1, noise_param2,
            maximum_total_weight, final_state_decision_method,
            noise_dispersion_based_on_weight, dispersion_factor,
            minimum_dispersion)
        self._initial_bb = initial_bb

    def tracker_score(self):
        """ Return the tracking score for this particle filter.
        It corresponds to the non-normalized sum of weights.
        """
        return self._pf._weight_sum

    def object_bb(self):
        return self._initial_bb.centered_on(*self.object_position())

    def object_position(self):
        return self._pf._final_state[:2]

    def init_particles(self, init_method='uniform', init_param1=None,
                       init_param2=None):
        """ Initialize all the particles.
        init_method must be either 'uniform' or 'gaussian'. This parameter
            indicates how the particles are initially spread in the state space
        init_param1 must be either None of a vector with self.num_states
            elements. If it is set as None, then it is initialized as
            self.particle_lower_bounds. If noise_type is gaussian, this
            parameter represents the means of the noise distribution, while if
            the noise_type is uniform, then it represents the lower bounds of
            the interval
        init_param2 is similar to init_param2. If it is None, it is set as
            self.particle_upper_bounds. When the noise_type is gaussian, it
            represents the standard deviations, while if it is uniform, it is
            the upper bounds of the interval
        """
        self._pf.init_particles(init_method, init_param1, init_param2)

    def update(self, img, particle_weight_function, mask=None):
        """ Updates the particle filter and returns the next predicted state and
        the total weights of the particles.
        """
        self._pf.update(particle_weight_function, img, mask)

    @property
    def num_states(self):
        return self._pf.num_states

    @property
    def particles(self):
        return self._pf.particles

    @property
    def weights(self):
        return self._pf.weights

    @property
    def weight_sum(self):
        return self._pf.weight_sum