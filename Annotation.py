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

from Rectangle import Rectangle


class Annotation(object):
    """ Represents one annotation file.

    Class variables:
    - entries: a list where each element represents one entry (line) of the file
    """
    def __init__(self, entries=None):
        self._entries = entries
        if entries is None:
            self._entries = []

    def add_entry(self, x, y, width, height, entry_index=-1):
        """ Adds or updates an entry of the list. If entry_index is negative,
        the entry is appended to the list, otherwise, the element at entry_index
        is replaced.
        """
        entry = Rectangle(x, y, width, height)
        if entry_index < 0:
            self._entries.append(entry)
        else:
            self._entries[entry_index] = entry

    def get_entry(self, entry_index):
        """ Returns one entry of the list """
        return self._entries[entry_index]

    def length(self):
        """ Returns the length of the list of entries. """
        return len(self._entries)

    def read(self, file_path, separator=','):
        """ Populates the entries list from the contents of a file.
        Each line of the file corresponds to one entry and it must follow the
        format:
        x,y,width,height
        where the separator (comma in the example above) may be changed
        """
        self._entries = []
        ann_file = open(file_path, 'r')
        content = ann_file.read()
        lines = content.replace('\r\n', '\n').split('\n')
        for line in lines:
            if len(line) > 0:
                vals = [int(float(x)) for x in line.split(separator)]
                entry = Rectangle(*vals)
                self._entries.append(entry)
        ann_file.close()

    def write(self, file_path, separator=','):
        """ Saves the entries list to a file.
        Each entry of the list will be saved as a separated line following the
        format:
        x,y,width,height
        where the separator (comma in the example above) may be changed
        """
        ann_file = open(file_path, 'w')

        for entry in self._entries:
            outLine = str(int(entry.x)) + separator + str(int(entry.y)) + separator + \
                str(int(entry.width)) + separator + str(int(entry.height)) + '\n'
            ann_file.write(outLine)

        ann_file.close()
        print('Wrote annotation file:', file_path)
