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

import os
from os import listdir
from Annotation import Annotation


class AnnotationSet(object):
    """ Represents one set of annotations. Typically a set is used
    when representing multiple objects simultaneously. In this case
    each annotation will correspond to a different object.
    """
    def __init__(self, video_name='', video_resolution=(0, 0), annotations=[]):
        self._video_name = video_name
        self._video_resolution = video_resolution
        self._num_objects = len(annotations)
        self._annotations = annotations

    def parse_config_file(self, file_path):
        """ Gets the contents of the config.ini file """
        config_file = open(file_path)
        lines = config_file.read().replace('\r\n', '\n').split('\n')
        num_found_params = 0
        i = 0
        video_name = ''
        video_resolution = (0, 0)
        prefix = ''
        while i < len(lines) and num_found_params < 3:
            if lines[i].startswith('video_name:'):
                video_name = lines[i].split(':')[1].replace(' ', '')
                num_found_params += 1
            elif lines[i].startswith('video_resolution:'):
                str_resol = lines[i].split(':')[1].replace(' ', '')
                video_resolution = tuple([int(x) for x in str_resol.split('x')])
                num_found_params += 1
            elif lines[i].startswith('annotation_files_prefix:'):
                prefix = lines[i].split(':')[1].replace(' ', '')
                num_found_params += 1
            i += 1

        return video_name, video_resolution, prefix

    def _read_annotations(self, directory_path, ann_file_prefix):
        """ Read all the annotations in the folder.  """
        file_names = [f for f in listdir(directory_path)]

        num_objects = 0
        for name in file_names:
            if name.startswith(ann_file_prefix):
                num_objects += 1

        annotations = [Annotation() for x in range(num_objects)]
        for name in file_names:
            if name.startswith(ann_file_prefix):
                extension_position = name.rfind('.')
                obj_index = int(name[len(ann_file_prefix):extension_position]) - 1
                annotations[obj_index].read(directory_path + name)

        return annotations

    def read_from_directory(self, directory_path):
        """ Fills the set by reading the contents from files in a directory.
        A file config.ini is required, with the following format:
        video_name: <name of the videos that generated the annotations>
        video_resolution: <frame width>x<frame height>
        annotation_files_prefix: <prefix, see below>

        Additionally, N other files named: <prefix>X.csv must exist, where N is
        the number of annotated objects,
        <prefix> is the name defined in the file config.ini and X is an integer
        in the interval [1, N]
        that corresponds to the object index (or label).
        """
        if directory_path[len(directory_path) - 1] != '/':
            directory_path += '/'
        video_name, video_resolution, prefix = \
            self.parse_config_file(directory_path + "config.ini")
        self._video_name = video_name
        self._video_resolution = video_resolution

        annotations = self._read_annotations(directory_path, prefix)
        self._num_objects = len(annotations)
        self._annotations = annotations

    def rescale_annotations(self, new_resolution):
        """ Changes all bounding boxes sizes and positions according to the
        new resolution.
        """
        scale_factor = float(new_resolution[0]) / self._video_resolution[0]
        self._video_resolution = new_resolution
        for i in range(self._num_objects):
            for j in range(self._annotations[i].length()):
                bb = self._annotations[i].get_entry(j)
                bb.scale_space(scale_factor)
                self._annotations[i].add_entry(bb.x, bb.y, bb.width, bb.height, j)

    def write(self, directory_path, dir_name_suffix=''):
        """ Create the directory: directory_path/self.video_name (without the
        extension) and the annotation files inside.
        """
        if directory_path[len(directory_path) - 1] != '/':
            directory_path += '/'
        dot_index = self._video_name.rfind('.')
        seq_name = self._video_name[:dot_index]
        if not os.path.exists(directory_path + seq_name + dir_name_suffix):
            os.makedirs(directory_path + seq_name + dir_name_suffix)

        config_file = open(directory_path + seq_name + dir_name_suffix +
                           '/config.ini', 'w')
        config_file.write('video_name: ' + self._video_name + '\n')
        config_file.write('video_resolution: ' + str(self._video_resolution[0]) +
                          'x' + str(self._video_resolution[1]) + '\n')
        config_file.write('annotation_files_prefix: ' + seq_name + '\n')
        config_file.close()

        for i in range(len(self._annotations)):
            self._annotations[i].write(directory_path + seq_name +
                                       dir_name_suffix + '/' +
                                       seq_name + str(i + 1) + '.csv')

    @property
    def annotations(self):
        return self._annotations

    @property
    def num_objects(self):
        return self._num_objects

    @property
    def video_name(self):
        return self._video_name

    @video_name.setter
    def video_name(self, val):
        self._video_name = val

    @property
    def video_resolution(self):
        return self._video_resolution
