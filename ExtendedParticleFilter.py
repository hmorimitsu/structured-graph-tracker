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

from ParticleFilter import ParticleFilter
import numpy as np


class ExtendedParticleFilter(ParticleFilter):
    """ Extends the ParticleFilter class by using an adaptive noise model whose
    variance is inversely proportional to the non-normalized sum of the weights
    of all particles.
    """

    def __init__(self, num_particles, num_states, dynamics_matrix,
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
        super(ExtendedParticleFilter, self).__init__(
            num_particles, num_states, dynamics_matrix, particle_lower_bounds,
            particle_upper_bounds, noise_type, noise_param1, noise_param2,
            final_state_decision_method)
        if maximum_total_weight > 0:
            self._maximum_total_weight = maximum_total_weight
        else:
            self._maximum_total_weight = num_particles
        self._noise_dispersion_based_on_weight = \
            noise_dispersion_based_on_weight
        self._dispersion_factor = dispersion_factor
        self._minimum_dispersion = minimum_dispersion

    def _propagate_particles(self):
        """ Applies dynamics and noise to all the particles. """
        dynamics_particles = np.dot(self._particles, self._dynamics_matrix)

        # The dispersion_ratio is multiplied by the original noise parameters to
        # obtain the adaptive noise
        dispersion_ratio = 1.0
        if self._noise_dispersion_based_on_weight:
            dispersion_ratio = self._dispersion_factor * \
                (1.0 - self._weight_sum / self._maximum_total_weight)

            if dispersion_ratio < self._minimum_dispersion:
                dispersion_ratio = self._minimum_dispersion

        if self._noise_type == 'uniform':
            noise = np.random.uniform(dispersion_ratio * self._noise_param1,
                                      dispersion_ratio * self._noise_param2,
                                      (self._num_particles, self._num_states))
        elif self._noise_type == 'gaussian':
            deviation = dispersion_ratio * np.array(self._noise_param2)
            noise = np.random.multivariate_normal(
                self._noise_param1, np.diag(deviation), self._num_particles)

        noise_particles = dynamics_particles + noise

        self._particles = noise_particles

#         """ Applies dynamics and noise to all the particles. """
#         for i in range(self._num_particles):
#             self._particles[i] = self._update_particle(self._particles[i])

    def update(self, weighting_function, *args):
        """ Updates all the particles by resampling, propagating and updating
        their weights.
        """
        self._resample_particles()
        self._propagate_particles()
        self._update_weights(weighting_function, *args)

        self.get_final_state()
