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
import sys
import Main
import Utils
from AnnotationSet import AnnotationSet

# Tableau 10 colors, obtained from:
# http://tableaufriction.blogspot.com.br/2012/11/finally-you-can-use-tableau-data-colors.html
colors = [(180, 119, 31), (14, 127, 255), (44, 160, 44), (40, 39, 214),
          (189, 103, 148), (75, 86, 140), (194, 119, 227), (127, 127, 127),
          (34, 189, 188), (207, 190, 23)]


def main(argv):
    if len(argv) < 1:
        print('Missing parameters, inform:')
        print('[video_path] (frame_width) (annotation1_path) (annotation1_name) (annotation2_path) (annotation2_name)...')
        print('if frame_width is set as zero, the original resolution will be used')
        sys.exit(-1)

    video_cap = Main.get_video_cap(argv[0])
    if Utils.is_cv2():
        num_frames = int(video_cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        original_width = int(video_cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        original_height = int(video_cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    elif Utils.is_cv3():
        num_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    desired_resolution = (original_width, original_height)
    desired_width = 0
    if len(argv) > 1:
        desired_width = int(argv[1])
    if desired_width > 0:
        scale_factor = float(desired_width) / original_width
        desired_resolution = (desired_width,
                              int(scale_factor * original_height))
    video_writer = cv2.VideoWriter('/home/henrique/graph_track_resutl.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), 30, desired_resolution)

    annotations = []
    names = []
    for iparam in range(2, len(argv), 2):
        ann_set = AnnotationSet()
        ann_set.read_from_directory(argv[iparam])
        if ann_set.video_resolution[0] != desired_resolution[0]:
            ann_set.rescale_annotations(desired_resolution)

        annotations.append(ann_set)
        names.append(argv[iparam + 1])

    print_instructions()

    cv2.namedWindow('output')
    iframe = 0
    auto_play = 1
    cv2.createTrackbar('Frames', 'output', 0, num_frames - 2, do_nothing)
    cv2.createTrackbar('Auto-play', 'output', 0, 1, do_nothing)
    frame_bgr = Main.get_video_frame(video_cap, desired_resolution)
    while True:
        trackbar_iframe = cv2.getTrackbarPos('Frames', 'output')
        auto_play = cv2.getTrackbarPos('Auto-play', 'output')
        if trackbar_iframe != iframe:
            if trackbar_iframe != iframe + 1:
                if Utils.is_cv2():
                    video_cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, trackbar_iframe)
                elif Utils.is_cv3():
                    video_cap.set(cv2.CAP_PROP_POS_FRAMES, trackbar_iframe)

            if Utils.is_cv2():
                iframe = int(video_cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
            elif Utils.is_cv3():
                iframe = int(video_cap.get(cv2.CAP_PROP_POS_FRAMES))

            frame_bgr = Main.get_video_frame(video_cap, desired_resolution)

        if auto_play == 1 and iframe < num_frames - 1:
            frame_bgr = Main.get_video_frame(video_cap, desired_resolution)
            if Utils.is_cv2():
                iframe = int(video_cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
            elif Utils.is_cv3():
                iframe = int(video_cap.get(cv2.CAP_PROP_POS_FRAMES))
            cv2.setTrackbarPos('Frames', 'output', iframe)

        draw_annotation_bbs(frame_bgr, annotations, iframe)
        draw_annotation_names(frame_bgr, names)

        cv2.imshow('output', frame_bgr)
        if auto_play == 1:
            video_writer.write(frame_bgr)

        key = cv2.waitKey(30)

        if key % 256 == 32:
            auto_play = (auto_play + 1) % 2
            cv2.setTrackbarPos('Auto-play', 'output', auto_play)
        elif key % 256 == ord('s'):
            cv2.imwrite(str(iframe) + '.png', frame_bgr)
            print('Saved screenshot to ' + str(iframe) + '.png')
        elif key % 256 == 27:
            break


def draw_annotation_bbs(frame_bgr, annotations, iframe):
    """ Draws the bouding boxes on the screen. """
    if len(annotations) > 0:
        global colors

        num_anns = len(annotations)
        for i in range(len(annotations) - 1, -1, -1):
            ann_set = annotations[i]
            for iobj in range(ann_set.num_objects):
                if ann_set.annotations[iobj].length() > iframe:
                    bb = ann_set.annotations[iobj].get_entry(iframe)
                    bb_color = colors[i % len(colors)]

                    cv2.rectangle(frame_bgr, Utils.float_tuple_to_int(bb.tl()),
                                  Utils.float_tuple_to_int(bb.br()),
                                  bb_color, 5)


def draw_annotation_names(frame_bgr, names):
    """ Writes the trackers names on the screen. """
    global colors
    if len(names) > 0:
        num_names = len(names)

        cv2.rectangle(frame_bgr, (10, frame_bgr.shape[0] - 15 - 30 * num_names),
                      (150, frame_bgr.shape[0] - 10),
                      (255, 255, 255), -1)
        cv2.rectangle(frame_bgr, (10, frame_bgr.shape[0] - 15 - 30 * num_names),
                      (150, frame_bgr.shape[0] - 10),
                      (0, 0, 0), 2)

        for i, name in enumerate(names):
            cv2.putText(frame_bgr, name, (15, frame_bgr.shape[0] - 15 - 30 * (num_names - 1 - i)),
                        cv2.FONT_HERSHEY_PLAIN, 2, colors[i % len(colors)], 2)


def print_instructions():
    print('The following keys can be used:')
    print('SPACE - Pause/resume the video')
    print('S - Save the frame being displayed')
    print('ESC - Close the video and exit')


def do_nothing(val):
    """ Just a filler function. """
    pass


if __name__ == '__main__':
    main(sys.argv[1:])
