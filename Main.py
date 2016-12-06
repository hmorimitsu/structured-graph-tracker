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

import argparse
import cv2
import math
import numpy as np
import sys
import time
import Utils
from GraphTracker import GraphTracker
from Histogram import Histogram
from Rectangle import Rectangle
from Annotation import Annotation
from AnnotationSet import AnnotationSet
if sys.version_info[0] == 2:
    import ConfigParser
elif sys.version_info[0] == 3:
    import configparser

# Tableau 10 colors, obtained from:
# http://tableaufriction.blogspot.com.br/2012/11/finally-you-can-use-tableau-data-colors.html
colors = [(180, 119, 31), (14, 127, 255), (44, 160, 44), (40, 39, 214),
          (189, 103, 148), (75, 86, 140), (194, 119, 227), (127, 127, 127),
          (34, 189, 188), (207, 190, 23)]
verbose_level = 1
display_screen = True
online_model = True


def main(argv):
    global verbose_level
    global display_screen
    global online_model

    # Read contents from the files
    args = parse_config_file('config.ini', 'pf.ini')
    verbose_level = args.verbose_level
    display_screen = args.display_screen

    # Model graph histogram parameters
    num_bins_dist = 25
    num_bins_angle = 18
    dist_range = (0, 1)
    angle_range = (0, 2 * math.pi)
    dist_interval = float(dist_range[1] - dist_range[0]) / num_bins_dist
    angle_interval = float(angle_range[1] - angle_range[0]) / num_bins_angle

    # Initialize histograms
    dist_vals = np.zeros(num_bins_dist, np.float32)
    angle_vals = np.zeros(num_bins_angle, np.float32)
    dist_labels = np.ones(num_bins_dist + 1, np.float32) * dist_interval
    dist_labels = np.cumsum(dist_labels) - dist_interval
    angle_labels = np.ones(num_bins_angle + 1, np.float32) * angle_interval
    angle_labels = np.cumsum(angle_labels) - angle_interval

    dist_hists = [[Histogram(dist_vals, dist_labels) for x in range(args.num_objects)] for x in range(args.num_objects)]
    angle_hists = [[Histogram(angle_vals, angle_labels) for x in range(args.num_objects)] for x in range(args.num_objects)]
    if args.use_graph and not online_model:
        dist_hists, angle_hists = read_histogram_models(
            args.structure_model, args.num_objects)

    # Get initial annotations
    initial_bbs, frame_interval, ann_video_resolution = \
        read_initial_annotations(args.initial_annotations, args.num_objects)
    initial_centroids = []
    initial_confidences = np.ones(len(initial_bbs))
    for bb in initial_bbs:
        initial_centroids.append(bb.centroid())

    # Rescale the frame
    desired_width_resolution = args.run_width_resolution
    video_cap = get_video_cap(args.input_video)
    frame_bgr = get_video_frame(video_cap)
    frame_width = frame_bgr.shape[1]
    if desired_width_resolution <= 0:
        desired_width_resolution = frame_width
    frame_scale_factor = float(desired_width_resolution) / frame_width
    desired_resolution = (int(desired_width_resolution),
                          int(frame_bgr.shape[0] * frame_scale_factor))
    frame_bgr = cv2.resize(frame_bgr, desired_resolution)

    if online_model:
        update_model_histograms(initial_centroids, initial_confidences,
                                dist_hists, angle_hists, args.adjacency_matrix,
                                desired_width_resolution)

    # Rescale the initial bb to the desired size and position
    ann_scale_factor = float(desired_width_resolution) / ann_video_resolution[0]
    rescale_space_rectangles(initial_bbs, ann_scale_factor)

    # Initialize the graph tracker
    graph_tracker = GraphTracker(args, dist_hists, angle_hists, initial_bbs)
    graph_tracker.init_object_classifiers(args, frame_bgr)
    graph_tracker.init_starting_trackers(args, initial_bbs, desired_resolution)

    iframe = frame_interval[0]
    if iframe > 1:
        if Utils.is_cv2():
            video_cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, iframe)
        elif Utils.is_cv3():
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, iframe)
    frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    if display_screen:
        cv2.namedWindow('output')
    start_time = time.time()

    if frame_interval[1] > 0:
        num_frames = frame_interval[1]
    else:
        if Utils.is_cv2():
            num_frames = int(video_cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        elif Utils.is_cv3():
            num_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize output annotation files
    out_ann_set = init_annotation_set(args.input_video, ann_video_resolution,
                                      initial_bbs, ann_scale_factor)

    # Main loop of the video
    while iframe < num_frames:
        if verbose_level >= 1:
            print('Frame %d' % iframe)
        # track with graphs
        if args.use_graph:
            candidate_trackers = graph_tracker.generate_candidate_trackers(
                args, desired_resolution)
            graph_tracker.add_good_candidates(candidate_trackers, frame_hsv)
            draw_trackers_candidates(frame_bgr, candidate_trackers)

        graph_tracker.update_object_trackers(frame_hsv)

        if display_screen:
            draw_candidates_bbs = True
            draw_trackers_bbs(frame_bgr, graph_tracker.object_trackers,
                              draw_candidates_bbs)
            if args.use_graph:
                draw_graph(frame_bgr, args.adjacency_matrix,
                           graph_tracker.object_trackers)

            cv2.imshow('output', frame_bgr)

        # Find the best candidates
        if args.use_graph:
            graph_tracker.optimize_global_tracking(frame_hsv)
            if verbose_level >= 1:
                print_object_trackers_stats(
                    graph_tracker, 'Object tracker stats')

        # Collect positions and scores from best candidates
        obj_points = []
        confidences = []
        for iobj in range(len(graph_tracker.object_trackers)):
            obj_points.append(
                graph_tracker.object_trackers[iobj][0].object_position())
            confidences.append(
                graph_tracker.object_trackers[iobj][0].feature_instant_score)

        if online_model:
            update_model_histograms(obj_points, confidences, dist_hists,
                                    angle_hists, args.adjacency_matrix,
                                    desired_width_resolution)

        # Write output for one frame
        add_annotations_to_set(out_ann_set, initial_bbs,
                               graph_tracker.object_trackers, ann_scale_factor)

        key = cv2.waitKey(1)

        if key % 256 == 27:
            break

        iframe += 1
        frame_bgr = get_video_frame(video_cap, desired_resolution)
        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    end_time = time.time()
    elapsed_time = end_time - start_time
    time_per_frame = elapsed_time / iframe
    fps = 1 / time_per_frame
    if verbose_level >= 0:
        print('Total time (s):', elapsed_time)
        print('Num frames:', iframe)
        print('FPS:', fps)

    # Output file name
    out_ann_dir_suffix = '_pf'
    if args.use_graph:
        out_ann_dir_suffix = '_graph_' + str(args.feature_weight) + 'fw_' + \
            str(args.structure_weight) + 'sw_' + \
            str(args.overlap_weight) + 'ow_' + \
            str(args.old_weight_factor) + 'of_' + \
            str(args.candidate_insertion_threshold) + 'it_' + \
            str(args.old_tracker_removal_threshold) + 'rt'
    out_ann_set.write(args.output_annotation_dir, out_ann_dir_suffix)


def add_annotations_to_set(ann_set, initial_bbs, object_trackers,
                           ann_scale_factor):
    """ Add the current results of the object trackers to an AnnotationSet. """
    for i in range(len(object_trackers)):
        rescaled_bb = initial_bbs[i].centered_on(
            *object_trackers[i][0].object_position())
        rescaled_bb.scale_space(1 / ann_scale_factor)
        ann_set.annotations[i].add_entry(*rescaled_bb.xywh())


def create_config_parser_arguments(parser):
    """ Initialize parameters for command line arguments. """
    parser.add_argument(
        '-iv', '--input_video',  type=Utils.convert_string_to_val,
        help='Path to the video')
    parser.add_argument(
        '-sm', '--structure_model', type=Utils.convert_string_to_val,
        help='Path to the structural model file')
    parser.add_argument(
        '-ia', '--initial_annotations', type=Utils.convert_string_to_val,
        help='Path to the file with the initial bounding boxes')
    parser.add_argument(
        '-oa', '--output_annotation_dir', type=Utils.convert_string_to_val,
        help='Path the folder where the output annotations will be saved')
    parser.add_argument(
        '-no', '--num_objects', type=Utils.convert_string_to_val,
        help='Number of objects to be tracked')
    parser.add_argument(
        '-am', '--adjacency_matrix',  type=Utils.convert_string_to_val,
        help='Adjacency matrix for the graph. Must be specified as a python ' +
        'list of lists')
    parser.add_argument(
        '-cm', '--candidates_matrix', type=Utils.convert_string_to_val,
        help='Candidates matrix for the graph. Must be specified as a python ' +
        'list of lists')
    parser.add_argument(
        '-dns', '--dist_noise_sigma', type=Utils.convert_string_to_val,
        help='Standard deviation of the distance sample for generating candidates.')
    parser.add_argument(
        '-ans', '--angle_noise_sigma', type=Utils.convert_string_to_val,
        help='Standard deviation of the angle sample for generating candidates.')
    parser.add_argument(
        '-fw', '--feature_weight', type=Utils.convert_string_to_val,
        help='Weight of the object appearance in the final score, must be a value' +
        'in the interval [0, 1]')
    parser.add_argument(
        '-sw', '--structure_weight', type=Utils.convert_string_to_val,
        help='Weight of the graph structure in the final score, must be a value' +
        'in the interval [0, 1]')
    parser.add_argument(
        '-ow', '--overlap_weight', type=Utils.convert_string_to_val,
        help='Weight of the overlpa penalty in the final score, must be a value' +
        'in the interval [0, 1]')
    parser.add_argument(
        '-of', '--old_weight_factor', type=Utils.convert_string_to_val,
        help='Weight of the old value for the temporal score.')
    parser.add_argument(
        '-it', '--candidate_insertion_threshold', type=Utils.convert_string_to_val,
        help='Threshold for accepting candidates.')
    parser.add_argument(
        '-rt', '--old_tracker_removal_threshold', type=Utils.convert_string_to_val,
        help='Threshold for removing non-significant trackers.')
    parser.add_argument(
        '-soo', '--same_object_overlap_dist_threshold',
        type=Utils.convert_string_to_val,
        help='Value to consider that two trackers of the same object are overlapping.')
    parser.add_argument(
        '-ug', '--use_graph', type=Utils.convert_string_to_val,
        help='Pass True of False to specify if the graphs would be used.')
    parser.add_argument(
        '-rwr', '--run_width_resolution', type=Utils.convert_string_to_val,
        help='Resize the video frames to this width. The height is scaled accordingly.')
    parser.add_argument(
        '-vl', '--verbose_level', type=Utils.convert_string_to_val,
        help='Must be 0, 1, 2 or 3. THe higher the value, more messages are printed.')
    parser.add_argument(
        '-ds', '--display_screen', type=Utils.convert_string_to_val,
        help='Pass True of False to specify if the results should be shown on the screen.')
    parser.add_argument(
        '-dg', '--draw_graph', type=Utils.convert_string_to_val,
        help='Pass True of False to specify if the graph should be drawn on the screen.')
    parser.add_argument(
        '-dob', '--draw_objects_bbs', type=Utils.convert_string_to_val,
        help='Pass True of False to specify if the trackers bounding boxes should be drawn on the screen.')
    parser.add_argument(
        '-dcb', '--draw_candidates_bbs', type=Utils.convert_string_to_val,
        help='Pass True of False to specify if if the candidate positions should be drawn on the screen.')


def create_pf_parser_arguments(parser):
    """ Initialize parameters for command line arguments. """
    parser.add_argument(
        '-np', '--num_particles', type=Utils.convert_string_to_val,
        help='Number of particles for each particle filter.')
    parser.add_argument(
        '-ns', '--num_states', type=Utils.convert_string_to_val,
        help='Number of states that represent the object.')
    parser.add_argument(
        '-dm', '--dynamics_matrix', type=Utils.convert_string_to_val,
        help='Dynamics matrix of the particle filter. Must be specified as a python ' +
        'list of lists')
    parser.add_argument(
        '-plb', '--particle_lower_bounds', type=Utils.convert_string_to_val,
        help='Lower bounds of the particle states. Must be a python list.')
    parser.add_argument(
        '-pub', '--particle_upper_bounds', type=Utils.convert_string_to_val,
        help='Upper bounds of the particle states. Must be a python list.')
    parser.add_argument(
        '-nt', '--noise_type', type=Utils.convert_string_to_val,
        help='Particle noise. Must be gaussian or uniform.')
    parser.add_argument(
        '-np1', '--noise_param1', type=Utils.convert_string_to_val,
        help='One parameter of the noise. Depends on noise type and must be a python list.')
    parser.add_argument(
        '-np2', '--noise_param2', type=Utils.convert_string_to_val,
        help='Another parameter of the noise. Depends on noise type and must be a python list.')
    parser.add_argument(
        '-fsd', '--final_state_decision_method',
        type=Utils.convert_string_to_val,
        help='Method to decide the state from the particles. Must be best, average or weighted_average.')
    parser.add_argument(
        '-mtw', '--maximum_total_weight', type=Utils.convert_string_to_val,
        help='Value used to adapt the spread on ExtendedParticleFilter.')
    parser.add_argument(
        '-ndw', '--noise_dispersion_based_on_weight',
        type=Utils.convert_string_to_val,
        help='Pass True of False to specify if the adaptive spread should be used.')
    parser.add_argument(
        '-df', '--dispersion_factor', type=Utils.convert_string_to_val,
        help='Value used to adapt the spread on ExtendedParticleFilter.')
    parser.add_argument(
        '-md', '--minimum_dispersion', type=Utils.convert_string_to_val,
        help='Value used to adapt the spread on ExtendedParticleFilter.')
    parser.add_argument(
        '-hc', '--channels', type=Utils.convert_string_to_val,
        help='A python list of the image channels to be used to compute the color histogram.')
    parser.add_argument(
        '-hm', '--mask', type=Utils.convert_string_to_val,
        help='A binary matrix mask to specify the region to compute the histogram from.')
    parser.add_argument(
        '-hnb', '--num_bins', type=Utils.convert_string_to_val,
        help='Number of bins used in the histogram.')
    parser.add_argument(
        '-hi', '--intervals', type=Utils.convert_string_to_val,
        help='A python list specifying the intervals of the histogram values.')
    parser.add_argument(
        '-pim', '--init_method', type=Utils.convert_string_to_val,
        help='How the particles are initialized. Must be uniform or gaussian.')
    parser.add_argument(
        '-pip1', '--init_param1', type=Utils.convert_string_to_val,
        help='One parameter of the particle initialization. Depends on init_method.')
    parser.add_argument(
        '-pip2', '--init_param2', type=Utils.convert_string_to_val,
        help='Another parameter of the particle initialization. Depends on init_method.')


def draw_graph(frame_bgr, adjacency_matrix, object_trackers):
    """ Draws the vertices and edges of the graph. """
    global colors
    for i in range(len(adjacency_matrix)):
        for j in range(len(adjacency_matrix[i])):
            if i != j and adjacency_matrix[i][j] > 0:
                pos1 = Utils.float_tuple_to_int(
                    object_trackers[i][0].object_position())
                pos2 = Utils.float_tuple_to_int(
                    object_trackers[j][0].object_position())
                cv2.line(frame_bgr, pos1, pos2, (255, 255, 255), 2)

    for i in range(len(object_trackers)):
        pos = Utils.float_tuple_to_int(
            object_trackers[i][0].object_position())
        cv2.circle(frame_bgr, pos, 7, colors[i % len(colors)], -1)


def draw_trackers_bbs(frame_bgr, object_trackers,
                      draw_candidates_bbs=False):
    """ Draw the bounding boxes of all the trackers. The best one is shown
    with a thicker line.
    """
    global colors
    for iobj in range(len(object_trackers)):
        for icand in range(len(object_trackers[iobj])):
            if not draw_candidates_bbs and icand > 0:
                break

            tracker_bb = object_trackers[iobj][icand].object_bb()
            thickness = 1
            if icand == 0:
                thickness = 5
            cv2.rectangle(frame_bgr, tracker_bb.tl(), tracker_bb.br(),
                          colors[iobj % len(colors)], thickness)


def draw_trackers_candidates(frame_bgr, candidate_trackers):
    """ Draw the bounding boxes of all the candidate trackers.
    """
    global colors
    for iobj in range(len(candidate_trackers)):
        for icand in range(len(candidate_trackers[iobj])):
            tracker_bb = candidate_trackers[iobj][icand].object_bb()
            cv2.rectangle(frame_bgr, tracker_bb.tl(), tracker_bb.br(),
                          colors[iobj % len(colors)], 1)


def draw_trackers_particles(frame_bgr, initial_bbs, graph_tracker):
    """ Draw the particle of all the best tracker of each object. """
    global colors

    for iobj in range(len(initial_bbs)):
        particles = graph_tracker.object_trackers[iobj][0]._tracker.particles
        for iparticle in range(len(particles)):
            particle_bb = initial_bbs[iobj].centered_on(*particles[iparticle])
            cv2.rectangle(frame_bgr, particle_bb.tl(), particle_bb.br(),
                          colors[iobj % len(colors)], 1)


def get_video_cap(video_path):
    """ Obtain the VideoCapture from the given video. """
    cap = cv2.VideoCapture(video_path)
    if (cap.isOpened() == False):
        print('Input video not found: ' + video_path)
        sys.exit(1)
    return cap


def get_video_frame(video_cap, desired_resolution=(0, 0)):
    """ Extract the next frame from video_cap and resize it, if necessary. """
    raw_frame = video_cap.read()[1]
    resized_frame = raw_frame
    if desired_resolution[0] > 0 and desired_resolution[1] > 0:
        resized_frame = cv2.resize(raw_frame, desired_resolution)
    return resized_frame


def init_annotation_set(video_path, video_resolution, initial_bbs,
                        ann_scale_factor):
    """ Creates and intializes an instance of AnnotationSet. """
    last_slash_index = video_path.rfind('/')
    video_name = video_path[last_slash_index + 1:]
    annotations2 = [Annotation() for x in range(len(initial_bbs))]
    ann_set = AnnotationSet(video_name, video_resolution, annotations2)
    for i in range(len(initial_bbs)):
        rescaled_bb = initial_bbs[i].clone()
        rescaled_bb.scale_space(1 / ann_scale_factor)
        ann_set.annotations[i].add_entry(*rescaled_bb.xywh())
    return ann_set


def parse_config_file(config_file, tracker_file, tracker_type='pf'):
    """ Read the config file. The file_type must be either
    'config' of 'pf'.
    """
    parser = argparse.ArgumentParser()
    if sys.version_info[0] == 2:
        config = ConfigParser.SafeConfigParser()
    elif sys.version_info[0] == 3:
        config = configparser.SafeConfigParser()
    config.read([config_file, tracker_file])

    create_config_parser_arguments(parser)
    if tracker_type == 'pf':
        create_pf_parser_arguments(parser)

    config_pairs = []
    for sec in config.sections():
        config_pairs += config.items(sec)
    defaults = dict(config_pairs)
    keys = defaults.keys()
    for k in keys:
        val = Utils.convert_string_to_val(defaults[k])
        defaults[k] = val

    parser.set_defaults(**defaults)

    return parser.parse_args()


def print_object_trackers_stats(graph_tracker, title):
    """ Prints the data about all the trackers. For debugging purposes. """
    print(title)
    print('iobj\ticand\tcentroid\t\ttemp_score\tinstant_score\t' +
          'feat_score\tstruct_score\toverlap_score\tchange_score')
    for i in range(len(graph_tracker.object_trackers)):
        for j in range(len(graph_tracker.object_trackers[i])):
            tracker = graph_tracker.object_trackers[i][j]
            print('%d\t%d\t(%06.2f, %06.2f)\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f' %
                  (i, j, tracker.object_position()[0],
                   tracker.object_position()[1],
                   tracker.temporal_score,
                   tracker.instant_score,
                   tracker.feature_instant_score,
                   tracker.structural_instant_score,
                   tracker.overlap_instant_score,
                   tracker.change_tracker_instant_score))


def read_histogram_models(model_path, num_objects,
                          dist_prefix='dist', angle_prefix='angle'):
    """ Read the histogram models from a file. The file must have a specific
    format, as follows:
    - one header line in the format: <dist_prefix or angle_prefix>_X_Y
    where X and Y are integer stating the indices of the pair of objects.
    - after the header, there must be N lines in the format:
    <label> <value>
    where N is the number of bins of the histogram (notice the white space
    between the two)
    There must not be any blank lines in the file (even at the end).
    """
    # Read the file
    model_file = open(model_path, 'r')
    lines = model_file.read().replace('\r\n', '\n').split('\n')
    if lines[len(lines) - 1] == '':
        lines = lines[:len(lines) - 1]

    dist_hists = [[None for x in range(num_objects)] for x in range(num_objects)]
    angle_hists = [[None for x in range(num_objects)] for x in range(num_objects)]

    # Read it line by liine
    i = 0
    while i < len(lines):
        if lines[i].replace(' ', '') == '':
            continue
        elif lines[i].find(dist_prefix) >= 0 or \
                lines[i].find(angle_prefix) >= 0:
            # Found one header, take the name and indices
            tokens = lines[i].split('_')
            name = tokens[0]
            iorigin = int(tokens[1])
            idest = int(tokens[2])

            # Read the histogram labels and values
            hist_labels = []
            hist_values = []
            for j in range(i + 1, len(lines)):
                tokens = lines[j].split(' ')
                if not Utils.is_number(tokens[0]):
                    i = j - 1
                    break
                hist_labels.append(float(tokens[0]))
                hist_values.append(float(tokens[1]))

            # Put the histogram in the respective list
            if name.find(dist_prefix) >= 0:
                dist_hists[iorigin][idest] = Histogram(
                    hist_values, hist_labels,
                    dist_prefix + '_' + str(iorigin) + '_' + str(idest))
            elif name.find(angle_prefix) >= 0:
                angle_hists[iorigin][idest] = Histogram(
                    hist_values, hist_labels,
                    angle_prefix + '_' + str(iorigin) + '_' + str(idest))
        i += 1

    model_file.close()
    return dist_hists, angle_hists


def read_initial_annotations(file_path, num_objects, separator=','):
    """ Read the file with the initial bouding boxes. The file must have a
    specific format:
    - one starting line:
    video_resolution: (<width>, <height>)
    - followed by another line:
    initial_bbs
    - afterwards there must be num_objects lines in the format:
    <x>,<y>,<width>,<height>
    where the values at the i-th line correspond to the bounding box of the
    i-th object
    """
    init_ann_file = open(file_path, 'r')
    lines = init_ann_file.read().replace('\r\n', '\n').split('\n')
    if lines[len(lines) - 1] == '':
        lines = lines[:len(lines) - 1]

    num_found_param = 0
    i = 0
    video_resolution = (0, 0)
    frame_interval = [0, 0]
    while i < len(lines) and num_found_param < 2:
        if lines[i].startswith('video_resolution:'):
            str_resol = lines[i].split(':')[1].replace(' ', '')
            video_resolution = tuple([int(x) for x in str_resol.split('x')])
            num_found_param += 1
        if lines[i].startswith('frames:'):
            str_resol = lines[i].split(':')[1].replace(' ', '')
            tokens = str_resol.split(',')
            frame_interval[0] = int(tokens[0])
            if Utils.is_number(tokens[1]):
                frame_interval[1] = int(tokens[1])

            num_found_param += 1
        i += 1

    while i < len(lines) and not lines[i].startswith('initial_bbs'):
        i += 1
    i += 1

    initial_bbs = []
    num_read_objects = 0
    while i < len(lines) and num_read_objects < num_objects:
        vals = [int(x) for x in lines[i].split(separator)]
        bb = Rectangle(*vals)
        initial_bbs.append(bb)
        num_read_objects += 1
        i += 1

    init_ann_file.close()
    return initial_bbs, frame_interval, video_resolution


def rescale_space_rectangles(rects_list, scale_factor):
    """ Calls the Rectangle.scale_space() function to a list of
    Rectangles.
    """
    for rect in rects_list:
        rect.scale_space(scale_factor)


def update_histogram(histogram, value, confidence):
    """ Update a histogram by adding the given value. The value is convolved
    according to its confidence. If the confidence is low, then it is spread
    in a wider range of histogram bins. """
    variance = int((1.0 - confidence) / 0.3)
    kernel = [1.0]
    if variance == 1:
        kernel = [0.3, 0.4, 0.3]
    elif variance == 2:
        kernel = [0.15, 0.2, 0.3, 0.2, 0.15]
    elif variance > 2:
        kernel = [0.1, 0.13, 0.17, 0.2, 0.17, 0.13, 0.1]
    kernel = np.array(kernel)
    bin_index = histogram.get_bin_index(value)
    histogram.add_value(bin_index, 1.0, kernel)


def update_model_histograms(obj_points, confidences, dist_hists,
                            angle_hists, adjacency_matrix, frame_width):
    """ Updates all the model histograms with the given values. """
    for iorigin in range(len(obj_points)):
        for idest in range(len(obj_points)):
            if iorigin != idest and adjacency_matrix[iorigin][idest] > 0:
                dist = Utils.compute_relative_distance(obj_points[iorigin], obj_points[idest], frame_width)
                angle = Utils.compute_angle(obj_points[idest], obj_points[iorigin])
                update_histogram(dist_hists[iorigin][idest], dist, confidences[iorigin])
                update_histogram(angle_hists[iorigin][idest], angle, confidences[iorigin])


if __name__ == '__main__':
    main(sys.argv[1:])
