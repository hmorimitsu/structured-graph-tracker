Structural Graph Tracker
------------------------

This code implements a method for tracking multiple object in videos using
structural information encoded in graphs and particle filters.

Copyright (c) 2016, Henrique Morimitsu,
University of Sao Paulo, Sao Paulo, Brazil

Contact: henriquem87@gmail.com

License
-------

This code is released under the BSD 3-clause license (see license.txt)

Citing Structural Graph Tracker
-------------------------------

If you consider this code useful for your research, please consider citing:

    @article{morimitsu2016exploring,
        author = {Henrique Morimitsu and Isabelle Bloch and Roberto M. Cesar-Jr.},
        title = {Exploring Structure for Long-Term Tracking of Multiple Objects in Sports Videos},
        journal = {Computer Vision and Image Understanding},
        year = {2016},
        doi = {http://dx.doi.org/10.1016/j.cviu.2016.12.003},
    }

Requirements
------------

Python: https://www.python.org/

Numpy: http://www.numpy.org/

OpenCV: http://opencv.org/

OpenCV must be installed with ffmpeg support in order to open videos.

This code was implemented and tested using Python 3.4 with OpenCV 3.0.0
but it should also work with Python 2.X and OpenCV 2.4.X

Usage
-----

The code is divided into two parts: tracking the objects and analyzing the
results. Most tracking parameters can be changed by modifying the files
config.ini and pf.ini

Tracking
--------

Before starting the tracking, you should update the file config.ini with your
configurations. Mainly, be sure that the following parameters point to the
correct files:
- input_video
- initial_annotations

You can also change the values directly on the terminal by specifying parameter
flags. In order to know the modifiable parameters, run the command:
> python Main.py -h

If the configuration is correct, then the tracking can be starter by running:
> python Main.py

You can stop the execution at any time by pressing the ESC key. The tracking
results will be saved on the folder specified in the output_annotation_dir
variable defined in config.ini.

Analyzing results
-----------------

You can compare the produced tracking results with a ground truth using the
TrackingEvaluator.py code. See more details about the computed measurements
in the comments section of file TrackingEvaluator.py.

In order to evaluate the results for the tt sample video, you should run the
command:

> python TrackingEvaluator.py outputAnnotations/ttd2_sample/ sequences/gt_ann/ttd2/ 2 2

Again, you can obtain more information about the parameters by simply calling
the code without parameters:

> python TrackingEvaluator.py

You can also hide/show the final header of the printed results and choose to
print or not the individual results for each frame by changing the flag
variables is_print_final_header and is_print_list directly on the source code
of TrackingEvaluator.py.

It is also possible to visualize the tracking results using the
ResultsVisualizator.py code. In order to visualize the ttd2_sample
results, you can run the following command:

> python ResultsVisualizator.py sequences/ttd2.mp4 0 outputAnnotations/ttd2_sample/ Sample

If you want to visualize multiple results simultaneously, simply add more
parameters, like:

> python ResultsVisualizator.py sequences/ttd2.mp4 0 outputAnnotations/ttd2_sample/ Sample sequences/gt_ann/ttd2/ GT

As always, you can obtain more information about the parameters by simply calling
the code without parameters:

> python ResultsVisualizator.py
