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
import random
import Utils
from ColorHistObjectClassifier import ColorHistObjectClassifier
from PFTracker import PFTracker
from SingleGraphTracker import SingleGraphTracker
from Rectangle import Rectangle


class GraphTracker(object):
    """ Tracks multiple objects using graphs. """
    def __init__(self, args, dist_hists, angle_hists, initial_bbs):
        self._dist_hists = dist_hists
        self._angle_hists = angle_hists
        self._initial_bbs = initial_bbs
        self._num_objects = args.num_objects
        self._adjacency_matrix = args.adjacency_matrix
        self._candidates_matrix = args.candidates_matrix
        self._dist_noise_sigma = args.dist_noise_sigma
        self._angle_noise_sigma = args.angle_noise_sigma
        self._feature_weight = args.feature_weight
        self._structure_weight = args.structure_weight
        self._overlap_weight = args.overlap_weight
        self._old_weight_factor = args.old_weight_factor
        self._candidate_insertion_threshold = args.candidate_insertion_threshold
        self._old_tracker_removal_threshold = args.old_tracker_removal_threshold
        self._same_object_overlap_threshold = \
            args.same_object_overlap_threshold
        self._verbose_level = args.verbose_level
        self._display_screen = args.display_screen
        self._object_classifiers = [None for x in range(args.num_objects)]
        self._object_trackers = [[] for x in range(args.num_objects)]

        self._best_combination = []
        self._score_best_combination = -99999

    def add_good_candidates(self, candidate_trackers, img_hsv):
        """ Adds candidates whose feature scores are above
        self.insertion_threshold.
        """
        for iobj in range(self._num_objects):
            for icand in range(len(candidate_trackers[iobj])):
                if not self.is_overlapping_chosen_trackers(
                        candidate_trackers[iobj][icand], self._object_trackers[iobj],
                        self._same_object_overlap_threshold):
                    candidate_trackers[iobj][icand].update_tracker(
                        img_hsv, self._object_classifiers[iobj].particle_weight)
                    feature_score = \
                        candidate_trackers[iobj][icand].tracker_feature_score()
                    if feature_score > self._candidate_insertion_threshold:
                        self._object_trackers[iobj].append(
                            candidate_trackers[iobj][icand])

    def compute_graph_score(self, vertices_indices, frame_hsv):
        """ Compute the average score of the vertices of the graph given
        by the combination of vertices.
        """
        total_score = 0.0
        for iobj in range(len(vertices_indices)):
            temporal_score = self.compute_one_vertex_score(
                iobj, vertices_indices, frame_hsv)[0]
            total_score += temporal_score

        total_score /= len(vertices_indices)
#         print(vertices_indices, total_score)
        return total_score

    def compute_one_vertex_feature_score(self, object_tracker):
        """ Compute and return the feature (or appearance) score of one
        vertex (tracker).
        """
        return object_tracker.tracker_feature_score()

    def compute_one_vertex_score(self, iobj, vertices_indices, frame_hsv):
        """ Computes the instant score of one vertex. """
        feature_score = self.compute_one_vertex_feature_score(
            self._object_trackers[iobj][vertices_indices[iobj]])
        structural_score = self.compute_one_vertex_structural_score(
            iobj, vertices_indices, self._dist_hists, self._angle_hists,
            self._adjacency_matrix, frame_hsv.shape[1])
        overlap_score = self.compute_one_vertex_overlap_score(
            iobj, vertices_indices, frame_hsv)
        change_tracker_score = 0.0
        if vertices_indices[iobj] > 0:
            change_tracker_score = 1.0
        temporal_score = self._object_trackers[iobj][vertices_indices[iobj]].compute_temporal_score(
            feature_score, structural_score, overlap_score, change_tracker_score)

        return temporal_score, feature_score, structural_score, overlap_score, change_tracker_score

    def compute_one_vertex_overlap_score(self, iobj, vertices_indices,
                                         frame_hsv):
        """ Compute and return the overlap score of one
        vertex (tracker).
        """
        overlap_score = 0.0
        for iobj2, icand2 in enumerate(vertices_indices):
            if iobj != iobj2:
                bb1 = self._object_trackers[iobj][vertices_indices[iobj]].object_bb()
                bb2 = self._object_trackers[iobj2][icand2].object_bb()
                interBB = bb1.intersection_region(bb2)
                if interBB.area() > 0:
                    hist1 = self._object_classifiers[iobj].compute_object_histogram(
                        frame_hsv, bb1, *self._object_classifiers[iobj].color_hist_params)
                    hist2 = self._object_classifiers[iobj2].compute_object_histogram(
                        frame_hsv, bb2, *self._object_classifiers[iobj].color_hist_params)
                    if Utils.is_cv2():
                        dist = cv2.compareHist(hist1, hist2,
                                               cv2.cv.CV_COMP_BHATTACHARYYA)
                    elif Utils.is_cv3():
                        dist = cv2.compareHist(hist1, hist2,
                                               cv2.HISTCMP_HELLINGER)
                    overlap_score += interBB.area() / bb1.area() * (1.0 - dist)

        return overlap_score

    def compute_one_vertex_structural_score(self, iobj, vertices_indices, dist_hists,
                                            angle_hists, adjacency_matrix,
                                            frame_width):
        """ Compute and return the structural score of one vertex (tracker). """
        obj_position = self._object_trackers[iobj][vertices_indices[iobj]].object_position()
        num_neighbors = 0
        total_dist_prob = 0.0
        total_angle_prob = 0.0
        # Find adjacent vertices and compute their distances and angles
        # probabilities
        for ineighbor in range(len(self._adjacency_matrix[iobj])):
            if ineighbor != iobj and adjacency_matrix[ineighbor][iobj] == 1:
                neighbor_position = \
                    self._object_trackers[ineighbor][vertices_indices[ineighbor]].object_position()
                dist = Utils.compute_relative_distance(
                    obj_position, neighbor_position, frame_width)
                angle = Utils.compute_angle(obj_position, neighbor_position)
                dist_prob = \
                    dist_hists[ineighbor][iobj].get_value_for_label(dist)
                angle_prob = \
                    angle_hists[ineighbor][iobj].get_value_for_label(angle)

                total_dist_prob += dist_prob
                total_angle_prob += angle_prob
                num_neighbors += 1
        # Normalize the score to interval [0, 1]
        vertex_score = \
            (total_dist_prob + total_angle_prob) / (2 * num_neighbors)
        return vertex_score

    def compute_position_from_origin(self, origin, dist, angle, frame_width):
        """ Compute a new coordinate based on an origin point, a distance and
        an angle.
        """
        return (origin[0] - math.cos(angle) * (dist * frame_width),
                origin[1] - math.sin(angle) * (dist * frame_width))

    def compute_trackers_changes_score(self, vertices_indices):
        """ Computes the number of tracker changes that this combination
        causes.
        """
        num_changes = 0
        for i in vertices_indices:
            if i != 0:
                num_changes += 1
        return float(num_changes) / len(vertices_indices)

    def compute_vertices_scores(self, frame_width):
        """ Compute and update the scores of all vertices (object trackers).
        """
        # Sort objects by their best scores
        score_indices = []
        for i in range(self._num_objects):
            score_indices.append((self._object_trackers[i][0].temporal_score, i))
        score_indices.sort()

        # Greedily compute scores by varying objects whose scores are lower
        for temporal_score, iobj in score_indices:
            vertices_indices = np.zeros(self._num_objects, np.uint8)
            for icand in range(len(self._object_trackers[iobj])):
                vertices_indices[iobj] = icand
                feature_score = self.compute_one_vertex_feature_score(
                    self._object_trackers[iobj][icand])
                structural_score = self.compute_one_vertex_structural_score(
                    iobj, vertices_indices, self._dist_hists, self._angle_hists,
                    self._adjacency_matrix, frame_width)
                self._object_trackers[iobj][icand].update_score(
                    structural_score, feature_score, 0, 0)

    def generate_candidate_positions(self, frame_width):
        """ Use the histogram models to generate new candidate centroid
        positions.
        """
        candidate_positions = [[] for x in range(self._num_objects)]

        for iorigin in range(len(self._candidates_matrix)):
            for iobj in range(len(self._candidates_matrix[iorigin])):
                if iorigin != iobj and \
                        self._candidates_matrix[iorigin][iobj] > 0:
                    for k in range(self._candidates_matrix[iorigin][iobj]):
                        position = \
                            self.generate_one_candidate_position(iorigin, iobj,
                                                                 frame_width)
                        candidate_positions[iobj].append(position)
        return candidate_positions

    def generate_candidate_trackers(self, pf_args, video_resolution):
        """ Use the histogram models to generate new candidate trackers. """
        candidate_positions = \
            self.generate_candidate_positions(video_resolution[0])
        candidate_trackers = [[] for x in range(self._num_objects)]

        for iobj in range(len(candidate_positions)):
            for icand in range(len(candidate_positions[iobj])):
                cand_tracker = self.init_one_pf_tracker(
                    pf_args, self._initial_bbs[iobj],
                    candidate_positions[iobj][icand], video_resolution)
                candidate_trackers[iobj].append(
                    SingleGraphTracker(
                        cand_tracker, self._old_weight_factor,
                        self._feature_weight, self._structure_weight,
                        self._overlap_weight))
        return candidate_trackers

    def generate_dist_angle(self, iorigin, iobj):
        """ Use the histogram models to generate a noisy distance and angle. """
        dist = self._dist_hists[iorigin][iobj].get_sampled_label()
        dist += self._dist_hists[iorigin][iobj].label_interval / 2
        angle = self._angle_hists[iorigin][iobj].get_sampled_label()
        angle += self._angle_hists[iorigin][iobj].label_interval / 2
        dist_noise = random.gauss(0.0, self._dist_noise_sigma)
        angle_noise = random.gauss(0.0, self._angle_noise_sigma)
        return dist + dist_noise, angle + angle_noise

    def generate_one_candidate_position(self, iorigin, iobj, frame_width):
        """ Use the histogram models to generate one new candidate centroid
        position.
        """
        origin_position = \
            self._object_trackers[iorigin][0].object_position()
        dist, angle = self.generate_dist_angle(iorigin, iobj)
        cand_position = self.compute_position_from_origin(origin_position,
                                                          dist, angle, frame_width)
        return cand_position

    def init_object_classifiers(self, classifier_args, img_bgr):
        """ Call the function to initialize all the object classifiers. """
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        self.init_color_hist_classifiers(classifier_args, img_hsv)

    def init_color_hist_classifiers(self, hist_args, img_hsv):
        """ Initialize a color histogram classifier for each object. """
        for i in range(len(self._object_classifiers)):
            color_classifier = self.init_one_color_hist_classifier(
                img_hsv, self._initial_bbs[i], hist_args.channels,
                hist_args.mask, hist_args.num_bins, hist_args.intervals)
            self._object_classifiers[i] = color_classifier

    def init_one_color_hist_classifier(
            self, hsv_img, bb, hist_channels, hist_mask, hist_num_bins,
            hist_intervals):
        """ Initialize one color histogram classifier and returns it. """
        color_classifier = ColorHistObjectClassifier(
            hsv_img, bb, hist_channels, hist_mask, hist_num_bins,
            hist_intervals)
        return color_classifier

    def init_starting_trackers(self, tracker_args, initial_bbs,
                               video_resolution):
        """ Call the function to initialize one single tracker
        for each object.
        """
        self._object_trackers = [[] for x in range(self._num_objects)]
        self.init_starting_pf_trackers(tracker_args, initial_bbs,
                                       video_resolution)

    def init_starting_pf_trackers(self, pf_args, initial_bbs, video_resolution):
        """ Initialize a particle filter for each object. """
        for i in range(self._num_objects):
            pf_tracker = self.init_one_pf_tracker(
                pf_args, initial_bbs[i], self._initial_bbs[i].centroid(),
                video_resolution)
            self._object_trackers[i].append(
                SingleGraphTracker(pf_tracker, self._old_weight_factor,
                                   self._feature_weight, self._structure_weight,
                                   self._overlap_weight, 1.0))

    def init_one_pf_tracker(self, pf_args, initial_bb, obj_centroid,
                            video_resolution):
        """ Initialize the general parameters of one particle filter and
        return it.
        """
        num_particles = pf_args.num_particles
        num_states = pf_args.num_states
        dynamics_matrix = pf_args.dynamics_matrix
        particle_lower_bounds = Utils.convert_None_number_list_to_list(
            pf_args.particle_lower_bounds, pf_args.num_states, 0.0)
        particle_upper_bounds = Utils.convert_None_number_list_to_list(
            pf_args.particle_upper_bounds, pf_args.num_states,
            video_resolution)
        noise_type = pf_args.noise_type
        noise_param1 = pf_args.noise_param1
        noise_param2 = pf_args.noise_param2
        maximum_total_weight = pf_args.maximum_total_weight
        final_state_decision_method = pf_args.final_state_decision_method
        noise_dispersion_based_on_weight = \
            pf_args.noise_dispersion_based_on_weight
        dispersion_factor = pf_args.dispersion_factor
        minimum_dispersion = pf_args.minimum_dispersion

        pf_tracker = PFTracker(
            initial_bb, num_particles, num_states, dynamics_matrix,
            particle_lower_bounds, particle_upper_bounds,
            noise_type, noise_param1, noise_param2,
            maximum_total_weight,
            final_state_decision_method,
            noise_dispersion_based_on_weight, dispersion_factor,
            minimum_dispersion)
        self.init_pf_particles(pf_tracker, pf_args, video_resolution,
                               obj_centroid)

        return pf_tracker

    def init_pf_particles(self, pf_tracker, pf_args, video_resolution,
                          centroid=(0, 0)):
        """ Initialize the particles of filter. """
        init_method = pf_args.init_method
        if init_method == 'uniform':
            init_param1 = Utils.convert_None_number_list_to_list(
                pf_args.init_param1, pf_args.num_states, 0)
            init_param2 = Utils.convert_None_number_list_to_list(
                pf_args.init_param2, pf_args.num_states,
                video_resolution)
        elif init_method == 'gaussian':
            init_param1 = Utils.convert_None_number_list_to_list(
                pf_args.init_param1, pf_args.num_states, centroid)
            init_param2 = Utils.convert_None_number_list_to_list(
                pf_args.init_param2, pf_args.num_states, 5)
        pf_tracker.init_particles(init_method, init_param1, init_param2)

    def is_overlapping_chosen_trackers(self, obj_tracker, chosen_trackers,
                                       overlap_threshold):
        """ Checks if a given object tracker overlaps with any of the previously
        chosen ones.
        """
        is_overlapping = False
        for i in range(len(chosen_trackers)):
            if self.is_trackers_overlapping(
                    obj_tracker, chosen_trackers[i],
                    overlap_threshold):
                is_overlapping = True
        return is_overlapping

    def is_trackers_overlapping(self, obj_tracker1, obj_tracker2,
                                overlap_threshold):
        """ Check if the tracker two is overlapping tracker one. """
        bb1 = obj_tracker1.object_bb()
        bb2 = obj_tracker2.object_bb()
        overlap_area = bb1.intersection_region(bb2).area()
        if bb1.area() > 0:
            overlap_area /= float(bb1.area())
        return (overlap_area > overlap_threshold)

    def optimize_global_tracking(self, frame_hsv):
        """ Improves the global tracking by creating all the graphs between
        the remaining trackers and trying to find the one that maximizes
        the graph score while minimizing the number of tracker changes
        (tries to keep the previous tracker).
        """
        self._score_best_combination = 0
        self._best_combination = []
        if self._verbose_level >= 2:
            print('combination\t\tscore\tgraph\tchanges')
#         self.optimize_global_tracking_rec([], frame_resolution)
        self.optimize_global_tracking_convergence(frame_hsv, 10)
        if self._verbose_level >= 1:
            print("Best combination\tscore")
            print(str(self._best_combination) + '\t\t' +
                  str(self._score_best_combination))

        self.update_trackers_scores(self._best_combination, frame_hsv)

        # Change this later, do not do like this
        for i, j in enumerate(self._best_combination):
            if j != 0:
                temp = self._object_trackers[i][0]
                self._object_trackers[i][0] = self._object_trackers[i][j]
                self._object_trackers[i][j] = temp

        if self._verbose_level >= 2:
            self.print_object_trackers_stats('Trackers - after global score')
        self.remove_non_significant_trackers()

    def optimize_global_tracking_convergence(
            self, frame_hsv, num_random_tries=0, max_iteration=10,
            min_diff=0.0):
        """ Iterative greedy method for trying to find the best graph.
        It is divided into two parts: in the first, the graph is initialized
        with the graph from last frame. In the second, random initializations
        are used to try to avoid local maxima.
        """
        # Try using old trackers as supports
        self._score_best_combination = 0.0
        self._best_combination = np.zeros(self._num_objects, np.uint8)
        i = 0
        diff = min_diff + 1
        while i < max_iteration and diff > min_diff:
            diff = 0
            # Sort objects by their best scores
            score_indices = []
            for iobj in range(self._num_objects):
                score_indices.append((self._object_trackers[iobj][self._best_combination[iobj]].temporal_score, iobj))
            score_indices.sort()

            for temporal_score, iobj in score_indices:
                vertices_indices = self._best_combination.copy()
                for icand in range(len(self._object_trackers[iobj])):
                    vertices_indices[iobj] = icand
                    graph_score = self.compute_graph_score(vertices_indices,
                                                           frame_hsv)
                    if graph_score > self._score_best_combination:
                        if self._verbose_level >= 2:
                            if self._score_best_combination > 0:
                                print('Changed global tracking, new vertices:')
                                print(vertices_indices)
                        diff = graph_score - self._score_best_combination
                        self._score_best_combination = graph_score
                        self._best_combination = vertices_indices.copy()
                    if self._verbose_level >= 3:
                        print(str(vertices_indices) + '\t\t%.2f' %
                              graph_score)
            i += 1

        # Random tries
        for itry in range(num_random_tries):
            i = 0
            best_score = 0.0
            best_vertices_indices = np.zeros(self._num_objects, np.uint8)
            for iobj in range(self._num_objects):
                num_cands = len(self._object_trackers[iobj])
                icand = random.randint(0, num_cands - 1)
                best_vertices_indices[iobj] = icand

            while i < max_iteration and diff > min_diff:
                diff = 0
                indices = np.arange(self._num_objects)
                np.random.shuffle(indices)
                for iobj in indices:
                    vertices_indices = best_vertices_indices.copy()
                    for icand in range(len(self._object_trackers[iobj])):
                        vertices_indices[iobj] = icand
                        graph_score = self.compute_graph_score(vertices_indices,
                                                               frame_hsv)
                        if graph_score > best_score:
                            if self._verbose_level >= 2:
                                if best_score > 0:
                                    print('Changed global tracking, new vertices:')
                                    print(vertices_indices)
                            diff = graph_score - best_score
                            best_score = graph_score
                            best_vertices_indices = vertices_indices
                        if self._verbose_level >= 3:
                            print(str(vertices_indices) + '\t\t%.2f' %
                                  graph_score)

            if best_score > self._score_best_combination:
                if self._verbose_level >= 2:
                    print('Random try found best combination')
                    print(best_vertices_indices)
                self._score_best_combination = best_score
                self._best_combination = best_vertices_indices

    def print_object_trackers_stats(self, title):
        """ Prints the data about all the trackers. For debugging purposes. """
        print(title)
        print('iobj\ticand\tcentroid\t\ttemp_score\tinstant_score\t' +
              'feat_score\tstruct_score\toverlap_score\tchange_score')
        for i in range(len(self._object_trackers)):
            for j in range(len(self._object_trackers[i])):
                tracker = self._object_trackers[i][j]
                print('%d\t%d\t(%06.2f, %06.2f)\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f' %
                      (i, j, tracker.object_position()[0],
                       tracker.object_position()[1],
                       tracker.temporal_score,
                       tracker.instant_score,
                       tracker.feature_instant_score,
                       tracker.structural_instant_score,
                       tracker.overlap_instant_score,
                       tracker.change_tracker_instant_score))

    def remove_overlapping_trackers(self, is_trackers_sorted=True):
        """ Remove object trackers that overlaps with others of the same
        object (inter-object overlap).
        """
        for iobj in range(len(self._object_trackers)):
            chosen_trackers = []
            if not is_trackers_sorted:
                self._object_trackers[iobj] = self.safe_sort_trackers(
                    self._object_trackers[iobj],
                    lambda tracker: tracker.temporal_score, True)
            for icand in range(len(self._object_trackers[iobj])):
                obj_tracker = self._object_trackers[iobj][icand]
                if not self.is_overlapping_chosen_trackers(
                        obj_tracker, chosen_trackers,
                        self._same_object_overlap_threshold):
                    chosen_trackers.append(obj_tracker)
            self._object_trackers[iobj] = chosen_trackers

    def remove_non_significant_trackers(self):
        """ Removes objects trackers that are not significant. This is done in
        two steps:
        - threshold by tehri temporal scores
        - removal of overlapping trackers
        """
        self.threshold_object_trackers()
        self.remove_overlapping_trackers()

    def safe_sort_trackers(self, trackers_list, sorting_key,
                           descending_order=False):
        """ Sort the list of trackers normally, but keeps the first element
        in the first place regardless of the criteria. This is useful to do not
        lose the tracker used in the previous iteration.
        """
        first_tracker = trackers_list[0]
        remaining_trackers = trackers_list[1:]
        remaining_trackers.sort(key=sorting_key, reverse=descending_order)
        return [first_tracker] + remaining_trackers

    def threshold_object_trackers(self):
        """ Remove trackers whose temporal scores are below
        self._insertion_threshold.
        """
        for iobj in range(len(self._object_trackers)):
            icand = 1
            self._object_trackers[iobj] = self.safe_sort_trackers(
                self._object_trackers[iobj],
                lambda tracker: tracker.temporal_score, True)
            while (icand < len(self._object_trackers[iobj])) and \
                    (self._object_trackers[iobj][icand].temporal_score >
                     self._old_tracker_removal_threshold):
                icand += 1
            self._object_trackers[iobj] = self._object_trackers[iobj][:icand]

    def update_object_trackers(self, img_hsv):
        """ Update the object trackers that existed from previous frames (excludes
        candidates added in the current frame). """
        for iobj in range(len(self._object_trackers)):
            for icand in range(len(self._object_trackers[iobj])):
                self._object_trackers[iobj][icand].update_tracker(
                    img_hsv, self._object_classifiers[iobj].particle_weight)

    def update_trackers_scores(self, best_vertices_indices, frame_hsv):
        """ Compute and change the scores of all the trackers according to
        the configuration provided by best_vertices_indices.
        """
        for iobj in range(len(best_vertices_indices)):
            vertices_update = list(best_vertices_indices)
            for icand in range(len(self._object_trackers[iobj])):
                vertices_update[iobj] = icand
                temporal_score, feature_score, structure_score, overlap_score, change_tracker_score = \
                    self.compute_one_vertex_score(iobj, vertices_update, frame_hsv)
                self._object_trackers[iobj][icand].update_scores(
                    feature_score, structure_score, overlap_score,
                    change_tracker_score)

    @property
    def object_trackers(self):
        return self._object_trackers
