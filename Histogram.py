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

import numpy as np
import random


class Histogram(object):
    """ Represents one histogram with labels and values. """
    def __init__(self, values, labels, title='histogram', is_normalized=False):
        self._title = title
        self._labels = np.array(labels)
        self._values = np.array(values)
        self._cumulated_values = np.cumsum(self._values)
        self._is_normalized = is_normalized
        self._values_total = np.sum(self._values)
        self._label_interval = 0
        if len(labels) > 1:
            self._label_interval = labels[1] - labels[0]

    def add_value(self, bin_index, value, kernel=[1.0]):
        """ Adds the value to a given bin. The value is convolved by the kernel
        before being added to the histogram.
        """
        conv_value = np.convolve(value, kernel)
        left = bin_index - int(len(conv_value) / 2)
        left_displacement = 0
        if left < 0:
            left_displacement = -left
            left = 0
        right = bin_index + int(len(conv_value) / 2) + 1
        right_displacement = 0
        if right >= len(self._values):
            right_displacement = right - len(self._values)
            right = len(self._values)
        if right_displacement == 0:
            self._values[left:right] += \
                conv_value[left_displacement:]
        else:
            self._values[left:right] += \
                conv_value[left_displacement:-right_displacement]
        self._values_total = np.sum(self._values)
        self._cumulated_values = np.cumsum(self._values)
        self._is_normalized = False

    def get_bin_index(self, label):
        """ Returns the index of the bin that includes label. """
        bin_index = min(len(self._values)-1, max(0, np.searchsorted(self._labels, label)-1))
        return bin_index

    def get_label(self, index):
        """ Returns the label at a given index. """
        return self._labels[index]

    def get_sampled_label(self):
        """ Randomly choose a label according to the histogram distribution,
        i.e., labels with higher values are more likely to be chosen.
        """
        cum_value = self._values_total * random.random()
        index = np.searchsorted(self._cumulated_values, cum_value)
        return self._labels[index]

    def get_value_for_label(self, label):
        """ Search and return the value of the bin that includes label. """
        bin_index = self.get_bin_index(label)
        return self._values[bin_index]

    def get_value(self, index):
        """ Returns the value at a given index. """
        return self._values[index]

    def normalize(self):
        """ Normalize the values vector to have sum equals to one. """
        if not self._is_normalized:
            self._values = self._values / np.sum(self._values)
            self._cumulated_values = np.cumsum(self._values)
            self._values_total = np.sum(self._values)
            self._is_normalized = True

    def num_bins(self):
        """ Returns the number of bins of the histogram. """
        return self._values.shape[0]

    def write(self, output_path, write_type='a'):
        """ Writes the output of one histogram to a file.
        write_type is used to specify whether the results should be appended to
        the file (write_type='a'), or rewrite the whole file (write_type='w').
        """
        output_file = open(output_path, write_type)
        output_file.write(self._title + '\n')
        for i in range(self.num_bins()):
            output_file.write(str(self._labels[i]) + ' ' +
                              str(self._values[i]) + '\n')
        output_file.close()

    @property
    def label_interval(self):
        return self._label_interval

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value):
        self._title = value
