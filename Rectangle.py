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


class Rectangle(object):
    """ Represents a rectangle. """

    def __init__(self, x, y, width, height):
        self._x = x
        self._y = y
        self._width = width
        self._height = height

    def __repr__(self):
        return str([self._x, self._y, self._width, self._height])

    def area(self):
        """ Return the are of the rectangle. """
        return self._width * self._height

    def bottom(self):
        """ Return the highest y coordinate. """
        return self._y + self._height

    def br(self):
        """ Returns the coordinate of the bottom-right corner. """
        return (int(self._x + self._width), int(self._y + self._height))

    def centered_on(self, x, y):
        """ Returns a rectangle with self.width and self.height
        centered on (x, y). This operation does not change the original
        rectangle, only returns a new one. """
        return Rectangle(x - self._width / 2, y - self._height / 2,
                         self._width, self._height)

    def centroid(self):
        """ Returns the coordinate of the centroid. """
        return (self._x + self._width / 2, self._y + self._height / 2)

    def clone(self):
        """ Return a copy of this object. """
        new_rect = Rectangle(self._x, self._y, self._width, self._height)
        return new_rect

    def intersection_region(self, other_rect):
        """ Return the rectangle representing the intersection with
        another rectangle.
        """
        inter_x1 = max(self._x, other_rect.x)
        inter_y1 = max(self._y, other_rect.y)
        inter_x2 = min(self.right(), other_rect.right())
        inter_y2 = min(self.bottom(), other_rect.bottom())
        inter_width = inter_x2 - inter_x1
        inter_height = inter_y2 - inter_y1
        if inter_width < 0 or inter_height < 0:
            return Rectangle(0, 0, 0, 0)
        else:
            return Rectangle(inter_x1, inter_y1, inter_width, inter_height)

    def is_inside(self, point):
        """ Check if a point is inside the rectangle. If the point is in the
        border, it is considered inside.
        """
        if (self._x <= point[0] <= self.right()) and \
                (self._y <= point[1] <= self.bottom()):
            return True
        return False

    def left(self):
        """ Return the lowest x coordinate. """
        return self._x

    def right(self):
        """ Return the highest x coordinate. """
        return self._x + self._width

    def scale(self, scale_factor):
        """ Multiply the width and height by width_scale_factor and
        height_scale_factor, respectively, while keeping the same centroid as
        before. This operation changes the original rectangle.
        """
        scaled_width = scale_factor * self._width
        scaled_height = scale_factor * self._height
        width_change = scaled_width - self._width
        height_change = scaled_height - self._height
        scaled_x = self._x - width_change / 2
        scaled_y = self._y - height_change / 2
        self._x = scaled_x
        self._y = scaled_y
        self._width = scaled_width
        self._height = scaled_height

    def scale_space(self, scale_factor):
        """ Changes both the coordinates and the size of the rectangle.
        First the centroid position (x, y) is translated to a new position:
        (x * width_scale_factor, y * height_scale_factor). Afterwards the
        rectangles are scaled by calling the self.scale() function.
        This operation changes the original rectangle.
        """
        new_centroid = (self.centroid()[0] * scale_factor,
                        self.centroid()[1] * scale_factor)
        self.translate(new_centroid)
        self.scale(scale_factor)

    def tl(self):
        """ Returns the coordinate of the top-left corner. """
        return (int(self._x), int(self._y))

    def tlbr(self):
        """ Returns a representation of the rectangle using the top-left and
        bottom-right corners.
        """
        return (self.tl(), self.br())

    def top(self):
        """ Return the lowest y coordinate. """
        return self._y

    def translate(self, new_centroid):
        """ Performs the geometrical translation of the rectangle.
        This operation changes the original rectangle.
        """
        x_displacement = new_centroid[0] - self.centroid()[0]
        y_displacement = new_centroid[1] - self.centroid()[1]
        self._x += x_displacement
        self._y += y_displacement

    def xywh(self):
        """ Returns a tuple representation containing the rectangle parameters. """
        return (self._x, self._y, self._width, self._height)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height
