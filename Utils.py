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

import ast
import cv2
import math

ESCAPE_KEY_VALUE = 0
SPACE_KEY_VALUE = 1


def convert_None_number_list_to_list(variable, list_length, default_value=0):
    """ Gets the content of a variable and converts it into a list.
    The variable must contain only None, a number or a list.
    If it is already a list, nothing is done. If variable is a number,
    then a list is created where all the elements are the number.
    If it is None, a list is created by filling it with default_value.
    default_value may be either a single number or a tuple or list.
    if it is a tuple or list, the final list is created by concatenating
    default_value until it reaches list_length.
    """
    if variable is None:
        if isinstance(default_value, (int, float)):
            return [default_value for x in range(list_length)]
        elif isinstance(default_value, (list, tuple)):
            ret_list = []
            for i in range(list_length):
                ret_list.append(default_value[i % len(default_value)])
            return ret_list
    elif isinstance(variable, list):
        return variable
    elif isinstance(variable, (int, float)):
        return [variable for x in range(list_length)]


def compute_angle(p, q):
    """ Compute the clockwise angle between the horizontal axis and
    the vector pq.
    """
    angle = math.atan2(q[1] - p[1], q[0] - p[0])
#     angle = math.atan2(p[1] - q[1], p[0] - q[0])
    if angle < 0:
        angle = 2 * math.pi + angle

    return angle


def compute_distance(p, q):
    """ Compute the Euclidean distance between two vectors. """
    dist = 0
    for xp, xq in zip(p, q):
        dist += math.pow(xp - xq, 2)
 
    return math.sqrt(dist)


def compute_relative_distance(p, q, reference):
    """ Compute the Euclidean distance between two vectors. """
    dist = 0
    for xp, xq in zip(p, q):
        dist += math.pow(xp - xq, 2)

    return math.sqrt(dist) / reference


def convert_string_to_val(confstr):
    """ Convert a string to a python literal. """
    if is_boolean(confstr) or is_list(confstr) or is_None(confstr) or \
            is_number(confstr) or is_tuple(confstr):
        return ast.literal_eval(confstr)
    else:
        return confstr


def float_tuple_to_int(float_tuple):
    """ Receives a tuple (or list) of float elements and returns a new tuple with
    int elements.
    """
    new_list = []
    for i in range(len(float_tuple)):
        new_list.append(int(float_tuple[i]))
    return tuple(new_list)


def get_file_name_from_path(file_path):
    """ Returns only the file name from a given path, i.e. removes everything
    before the last slash character and everything after the last dot
    character.
    """
    file_name = file_path
    slash_position = file_name.rfind('/')
    dot_position = file_name.rfind('.')
    if slash_position >= 0:
        file_name = file_name[slash_position + 1:]
    if dot_position >= 0:
        file_name = file_name[:dot_position]
    return file_name


def is_cv2():
    """ Checks if OpenCV version 2.X is installed. """
    return cv2.__version__.startswith('2.')


def is_cv3():
    """ Checks if OpenCV version 3.X is installed. """
    return cv2.__version__.startswith('3.')


def is_boolean(str_seq):
    """ Checks if a given string is a boolean value. """
    if str_seq.lower() == 'true' or str_seq.lower() == 'false':
        return True
    return False


def is_list(str_seq):
    """ Checks if a given string is a list. """
    if str_seq.startswith('[') and str_seq.endswith(']'):
        return True
    return False


def is_None(str_seq):
    """ Checks if a given string is None. """
    if str_seq.lower() == 'none':
        return True
    return False


def is_number(str_seq):
    """ Checks if a given string is a number. """
    try:
        float(str_seq)
        return True
    except ValueError:
        return False


def is_tuple(str_seq):
    """ Checks if a given string is a tuple. """
    if str_seq.startswith('(') and str_seq.endswith(')'):
        return True
    return False
