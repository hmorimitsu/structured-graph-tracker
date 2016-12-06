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

import math


class SingleGraphTracker(object):
    """ Represents one vertex (tracker) of the graph. """
    def __init__(self, tracker, old_weight_factor, feature_weight, structure_weight,
                 overlap_weight, temporal_score=0.0, feature_instant_score=0.0,
                 structural_instant_score=0.0, overlap_instant_score=0.0,
                 change_tracker_instant_score=0.0):
        self._tracker = tracker
        self._old_weight_factor = old_weight_factor
        self._feature_weight = feature_weight
        self._structure_weight = structure_weight
        self._overlap_weight = overlap_weight
        self._temporal_score = temporal_score
        self._feature_instant_score = feature_instant_score
        self._structural_instant_score = structural_instant_score
        self._change_tracker_instant_score = change_tracker_instant_score
        self._overlap_instant_score = overlap_instant_score
        self._instant_score = self.compute_total_instant_score(
            feature_instant_score, structural_instant_score,
            change_tracker_instant_score, overlap_instant_score)

    def compute_temporal_score(self, feature_instant_score, structural_instant_score,
                               overlap_instant_score, change_tracker_instant_score):
        """ Compute the temporal score by linearly combining
        self._temporal_score with instant_score weighted by
        self._old_weight_factor.
        This function does not change the internal variables.
        """
        instant_score = self.compute_total_instant_score(
            feature_instant_score, structural_instant_score,
            overlap_instant_score, change_tracker_instant_score)
        return self._old_weight_factor * \
            self._temporal_score + instant_score

    def compute_total_instant_score(
            self, feature_instant_score, structural_instant_score,
            overlap_instant_score, change_tracker_instant_score):
        """ Compute the instant score by linearly combining
        feature_instant_score with structural_instant_score weighted by
        self._structure_weight.
        This function does not change the internal variables.
        """
        return self._feature_weight * feature_instant_score + \
            self._structure_weight * structural_instant_score - \
            self._overlap_weight * overlap_instant_score - \
            (1 - self._feature_weight - self._structure_weight -
             self._overlap_weight) * change_tracker_instant_score

    def tracker_feature_score(self):
        """ Compute and return the instant score of the tracker.
        This function does not change the internal variables.
        """
        return 1 - math.exp(-2 * self._tracker.tracker_score())

    def object_bb(self):
        """ Returns the bounding box of the object. """
        return self._tracker.object_bb()

    def object_position(self):
        """ Returns the (x, y) coordinate of the centroid of the object. """
        return self._tracker.object_position()

    def update_scores(self, feature_instant_score, structural_instant_score,
                      overlap_instant_score, change_tracker_instant_score):
        """ Replace the instant score and recompute the temporal
        using the old weight factor.
        """
        self._feature_instant_score = feature_instant_score
        self._structural_instant_score = structural_instant_score
        self._overlap_instant_score = overlap_instant_score
        self._change_tracker_instant_score = change_tracker_instant_score
        self._instant_score = self.compute_total_instant_score(
            feature_instant_score, structural_instant_score,
            overlap_instant_score, change_tracker_instant_score)
        self._temporal_score = self.compute_temporal_score(
            feature_instant_score, structural_instant_score,
            overlap_instant_score, change_tracker_instant_score)

    def update_tracker(self, img, *args):
        """ Update the tracker for the next frame. """
        self._tracker.update(img, *args)

    @property
    def change_tracker_instant_score(self):
        return self._change_tracker_instant_score

    @property
    def feature_instant_score(self):
        return self._feature_instant_score

    @property
    def instant_score(self):
        return self._instant_score

    @property
    def old_weight_factor(self):
        return self._old_weight_factor

    @property
    def overlap_instant_score(self):
        return self._overlap_instant_score

    @property
    def structural_instant_score(self):
        return self._structural_instant_score

    @property
    def structure_weight(self):
        return self._structure_weight

    @property
    def temporal_score(self):
        return self._temporal_score

    @property
    def tracker(self):
        return self._tracker