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
import numpy as np
import sys
import Utils
from AnnotationSet import AnnotationSet

""" Compare an Annotation Set (test results) with another (groundtruth) and
compute the following quantitative measurements:
CDIS: Center distance: a stability measurement that computes how much the centroid
      of the same object moves between two consecutive frames
CERR: Center error: Euclidean distance between the centroids of the bounding boxes
GINT: Global intersection: the overall intersection proportion between all the
      test boxes with the groundtruth
HITR: Hit ratio: a result is considered a hit when the centroid of the test box
      is inside the grountruth one
OBJI: Object intersection: the sum of the intersection proportion between a test
      box with its respective groundtruth

Besides, two of those measurements are relaxed specifically deal with tracking
players in sports team matches. These measurements disregard the
swapping of boxes between players of the same team:
CERT: Center error team: same as CERR but for teams
HITT: Hit ration team: same as HITR but for teams
It is assumed that, if there are n teams with m players each, the first
n * m objects are the players. Besides, it is also assumed that the first
m consecutive objects are from team 1, the next m from team 2 an so on.

All the measurements are scaled by dividing them by the total number of
processed objects (typically the number of frames multiplied by the number of
objects, if no holes exist).
"""

# Used for finding all combination of pairs in function find_pairs_combination
best_pairs_score = 0.0


def main(argv):
    if len(argv) < 2:
        print('Missing parameters, inform:')
        print('[test_ann_folder] [groudtruth_ann_folder] (num_teams) (num_team_players)')
        print('The last two arguments are optional and they are only used for computing' +
              'CERT and HITT (check the code documentation).')
        sys.exit(-1)
    # If set as True, prints all the values obtained at each frame
    is_print_final_header = True
    is_print_list = False

    num_teams = 0
    num_team_players = 0
    if len(argv) > 3:
        num_teams = int(argv[2])
        num_team_players = int(argv[3])

    test_ann_set = AnnotationSet()
    gt_ann_set = AnnotationSet()
    test_ann_set.read_from_directory(argv[0])
    gt_ann_set.read_from_directory(argv[1])

    if (test_ann_set.video_resolution[0] != gt_ann_set.video_resolution[0]):
        test_ann_set.rescale_annotations(gt_ann_set.video_resolution)

    video_resolution = test_ann_set.video_resolution

    # General measurements for any kind of video
    total_cdis = 0.0
    total_cerr = 0.0
    total_gint = 0.0
    total_hitr = 0.0
    total_obji = 0.0
    # Measurements specific for double team sports videos
    # It is assumed the pairs of objects (1, 2) and (3, 4) are players
    # of the same team
    total_cert = 0.0
    total_hitt = 0.0

    total_frames_processed = 0
    total_objects_processed = 0
    num_objects = min(test_ann_set.num_objects, gt_ann_set.num_objects)

    max_num_frames = 0
    for iobj in range(num_objects):
        num_frames = max(test_ann_set.annotations[iobj].length(),
                         gt_ann_set.annotations[iobj].length())
        max_num_frames = max(max_num_frames, num_frames)

    if is_print_list:
        print('Frame\tCDIS\tCERR\tGINT\tHITR\tOBJI\tCERT\tHITT')
    for iframe in range(max_num_frames):
        frame_cdis = 0.0
        frame_cerr = 0.0
        frame_gint = 0.0
        frame_hitr = 0.0
        frame_obji = 0.0
        frame_cert = 0.0
        frame_hitt = 0.0

        all_testBBs_in_frame = []
        all_gtBBs_in_frame = []
        objects_processed = 0
        for iobj in range(num_objects):
            num_frames = min(test_ann_set.annotations[iobj].length(),
                             gt_ann_set.annotations[iobj].length())
            if iframe < num_frames:
                if iobj == 0:
                    total_frames_processed += 1
                testBB = test_ann_set.annotations[iobj].get_entry(iframe)
                gtBB = gt_ann_set.annotations[iobj].get_entry(iframe)
                if testBB.area() > 0 and gtBB.area() > 0:
                    all_testBBs_in_frame.append(testBB)
                    all_gtBBs_in_frame.append(gtBB)

                    if iframe > 0:
                        old_testBB = \
                            test_ann_set.annotations[iobj].get_entry(iframe - 1)
                        frame_cdis += compute_cdis(old_testBB, testBB)
                    frame_cerr += compute_cerr(testBB, gtBB)
                    frame_hitr += compute_hitr(testBB, gtBB)
                    frame_obji += compute_obji(testBB, gtBB)

                    # Compute CERT and HITT
                    if (num_objects >= num_teams * num_team_players) and \
                            (iobj < num_teams * num_team_players) and \
                            (iobj % num_team_players == 0):
                        team_testBBs = [testBB]
                        team_gtBBs = [gtBB]
                        for i_team_obj in range(1, num_team_players):
                            t_testBB = test_ann_set.annotations[
                                iobj + i_team_obj].get_entry(iframe)
                            t_gtBB = gt_ann_set.annotations[
                                iobj + i_team_obj].get_entry(iframe)
                            if t_testBB.area() > 0 and t_gtBB.area() > 0:
                                team_testBBs.append(t_testBB)
                                team_gtBBs.append(t_gtBB)
                        frame_cert += compute_cert(team_testBBs, team_gtBBs)
                        frame_hitt += compute_hitt(team_testBBs, team_gtBBs)
                    elif iobj >= num_teams * num_team_players:
                        frame_cert += compute_cerr(testBB, gtBB)
                        frame_hitt += compute_hitr(testBB, gtBB)

                    objects_processed += 1
        frame_gint += compute_gint(all_testBBs_in_frame, all_gtBBs_in_frame,
                                   video_resolution)

        total_cdis += frame_cdis
        total_cerr += frame_cerr
        total_gint += frame_gint
        total_hitr += frame_hitr
        total_obji += frame_obji
        total_cert += frame_cert
        total_hitt += frame_hitt

        total_objects_processed += objects_processed

        if is_print_list:
            print_list_results(iframe, frame_cdis, frame_cerr, frame_gint,
                               frame_hitr, frame_obji, frame_cert, frame_hitt,
                               objects_processed)

    print_final_results(total_cdis, total_cerr, total_gint, total_hitr,
                        total_obji, total_cert, total_hitt,
                        total_objects_processed, total_frames_processed,
                        is_print_final_header)


def compute_cdis(old_testBB, curr_testBB):
    """ Compute the CDIS measurement. """
    dist = Utils.compute_distance(old_testBB.centroid(), curr_testBB.centroid())
    return dist


def compute_cerr(testBB, gtBB):
    """ Compute the CERR measurement. """
    dist = Utils.compute_distance(testBB.centroid(), gtBB.centroid())
    return dist


def compute_cert(team_testBBs, team_gtBBs):
    """ Compute the CERT measurement. """
    global best_pairs_score
    find_best_pairs_bbs(team_testBBs, team_gtBBs, 'cert')
    return best_pairs_score


def compute_gint(all_testBBs_in_frame, all_gtBBs_in_frame, video_resolution):
    """ Compute the GINT measurement. """
    test_img = np.zeros((video_resolution[1], video_resolution[0]), np.uint8)
    gt_img = np.zeros((video_resolution[1], video_resolution[0]), np.uint8)
    for iobj in range(len(all_testBBs_in_frame)):
        testBB = all_testBBs_in_frame[iobj]
        gtBB = all_gtBBs_in_frame[iobj]
        cv2.rectangle(test_img, Utils.float_tuple_to_int(testBB.tl()),
                      Utils.float_tuple_to_int(testBB.br()), (255), -1)
        cv2.rectangle(gt_img, Utils.float_tuple_to_int(gtBB.tl()),
                      Utils.float_tuple_to_int(gtBB.br()), (255), -1)

    inter_img = cv2.bitwise_and(test_img, gt_img)
    test_area = cv2.sumElems(test_img)[0]
    gt_area = cv2.sumElems(gt_img)[0]
    max_area = max(test_area, gt_area)

    if max_area == 0:
        return 0.0

    inter_area = cv2.sumElems(inter_img)[0]
    norm_inter_area = inter_area / float(max_area)

    return norm_inter_area


def compute_hitr(testBB, gtBB):
    """ Compute the HITR measurement. """
    if gtBB.is_inside(testBB.centroid()):
        return 1.0
    return 0.0


def compute_hitt(team_testBBs, team_gtBBs):
    """ Compute the HITT measurement. """
    global best_pairs_score
    find_best_pairs_bbs(team_testBBs, team_gtBBs, 'hitt')
    return best_pairs_score


def compute_obji(testBB, gtBB):
    """ Compute the OBJI measurement. """
    inter_rect = testBB.intersection_region(gtBB)
    inter_area = inter_rect.area()
    max_area = max(testBB.area(), gtBB.area())
    return float(inter_area) / max_area


def find_best_pairs_bbs(testBBs, gtBBs, measurement):
    """ Find the best combination pairs (testBB, gtBB) that minimize CERR of
    that maximize HITR, depending on the measurement parameter.
    measurement must be either 'cert' of 'hitt'
    """
    global best_pairs_score
    if measurement == 'cert':
        best_pairs_score = 9999999.9
    if measurement == 'hitt':
        best_pairs_score = 0.0

    num_objs = len(testBBs)
    used_test_objs = np.zeros((num_objs), np.bool)
    used_gt_objs = np.zeros((num_objs), np.bool)
    find_pairs_combination([], used_test_objs, used_gt_objs, measurement,
                           testBBs, gtBBs)


def find_pairs_combination(pairs, used_test_objs, used_gt_objs, measurement,
                           testBBs, gtBBs):
    """ Recursive part of find_best_pairs_bbs function. Generate all the
    combination of pairs (testBB, gtBB) and test.
    """
    global best_pairs_score

    num_objs = len(used_test_objs)
    if len(pairs) == num_objs:
        # Once the list of pairs is ready, process
        if measurement == 'cert':
            total_cerr = 0.0
            for p in pairs:
                total_cerr += compute_cerr(testBBs[p[0]], gtBBs[p[1]])
            if total_cerr < best_pairs_score:
                best_pairs_score = total_cerr
        elif measurement == 'hitt':
            total_hitt = 0.0
            for p in pairs:
                total_hitt += compute_hitr(testBBs[p[0]], gtBBs[p[1]])
            if total_hitt > best_pairs_score:
                best_pairs_score = total_hitt
    else:
        # Recursion part, generate new valid pairs
        for itest_obj in range(num_objs):
            for igt_obj in range(num_objs):
                if not used_test_objs[itest_obj] and \
                        not used_gt_objs[igt_obj]:
                    new_used_test_objs = used_test_objs.copy()
                    new_used_gt_objs = used_gt_objs.copy()
                    new_used_test_objs[itest_obj] = True
                    new_used_gt_objs[igt_obj] = True
                    new_pairs = list(pairs)
                    new_pairs.append((itest_obj, igt_obj))

                    find_pairs_combination(
                        new_pairs, new_used_test_objs, new_used_gt_objs,
                        measurement, testBBs, gtBBs)


def print_list_results(iframe, total_cdis, total_cerr, total_gint, total_hitr,
                       total_obji, total_cert, total_hitt, objects_processed):
    """ Print the results obtained in a single frame. """
    norm_total_cdis = total_cdis
    norm_total_cerr = total_cerr
    norm_total_gint = total_gint
    norm_total_hitr = total_hitr
    norm_total_obji = total_obji
    norm_total_cert = total_cert
    norm_total_hitt = total_hitt

    if objects_processed > 0:
        norm_total_cdis = total_cdis / objects_processed
        norm_total_cerr = total_cerr / objects_processed
        norm_total_gint = total_gint
        norm_total_hitr = total_hitr / objects_processed
        norm_total_obji = total_obji / objects_processed
        norm_total_cert = total_cert / objects_processed
        norm_total_hitt = total_hitt / objects_processed
    print('%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f' %
          (iframe, norm_total_cdis, norm_total_cerr, norm_total_gint,
           norm_total_hitr, norm_total_obji, norm_total_cert,
           norm_total_hitt))


def print_final_results(total_cdis, total_cerr, total_gint, total_hitr,
                        total_obji, total_cert, total_hitt,
                        total_objects_processed, total_frames_processed,
                        is_print_final_header = True):
    """ Print the results obtained for the whole video. """
    norm_total_cdis = total_cdis
    norm_total_cerr = total_cerr
    norm_total_gint = total_gint
    norm_total_hitr = total_hitr
    norm_total_obji = total_obji
    norm_total_cert = total_cert
    norm_total_hitt = total_hitt

    if total_objects_processed > 0:
        norm_total_cdis = total_cdis / total_objects_processed
        norm_total_cerr = total_cerr / total_objects_processed
        norm_total_gint = total_gint / total_frames_processed
        norm_total_hitr = total_hitr / total_objects_processed
        norm_total_obji = total_obji / total_objects_processed
        norm_total_cert = total_cert / total_objects_processed
        norm_total_hitt = total_hitt / total_objects_processed
    if is_print_final_header:
        print('Final results')
        print('\tCDIS\tCERR\tGINT\tHITR\tOBJI\tCERT\tHITT')
    print('\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f' %
          (norm_total_cdis, norm_total_cerr, norm_total_gint,
           norm_total_hitr, norm_total_obji, norm_total_cert,
           norm_total_hitt))

if __name__ == '__main__':
    main(sys.argv[1:])
