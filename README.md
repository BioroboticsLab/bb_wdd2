Waggle Dance Detector
=====================

This repository contains the current state of the BeesBook waggle dance detection software.
Since its publication in 2017 [Wario et al., 2017](https://doi.org/10.1371/journal.pone.0188626), most of the software has been rewritten and improved. The original software can be found here: [bb_wdd](https://github.com/BioroboticsLab/bb_wdd/).

About
-----

The WDD can read either video files or a camera stream. Supported cameras are everything that OpenCV supports. Additionally, Flea3 cameras are supported directly via the PySpin SDK or the PyCapture2 SDK (legacy).

The WDD processes the input live (for cameras) or as fast as possible (for videos) and saves all detections with short video snippets (as images) and metadata to the filesystem.
It can be integrated into other processes (e.g. automatic experimental control) with an IPC interface over which all detections will be sent as well.

The WDD is a command line application. This means that there is no graphical user interface.
There are some parameters that need to be set when launching the WDD, such as the FPS, the size of a bee in the video stream (in pixels) and the sensitivity.

The sensitivity is controlled by a number of hyperparameters. If you have an annotated video file, you can let the WDD automatically figure out the optimal parameter settings.
The software should run at least at 60 FPS (so probably not on a RaspberryPi); the WDD is flexible with respect to the image resolution.

Installation
------------

The package can be installed from the repository here via pip.
`pip install git+https://github.com/BioroboticsLab/bb_wdd2.git`

Use
---

To list all possible arguments, execute `bb_wdd --help`.

### Examples:

#### Apply the WDD with default sensitivity settings to a video file.

The parameters in this example that define the size of the video, the FPS and the bee length (in pixels) have to be set always.
The parameter `no_warmup` disabled the default behavior of waiting until frames can be read at the given FPS from the image stream, which might be helpful for live capture.
`video_device` can be a file or an OpenCV camera identifier. The default `capture_type` is OpenCV.

The `subsample` arguments here scales the images down by a factor of 8 before processing. This greatly increases the processing speed. A certain amount of subsampling might even improve the detections, because it reduces pixel luminance noise.
This is the main argument to control the speed of the WDD.

The `cam_identifier` is arbitrary and will be saved in the metadata of the waggles. It can be a camera identifier, side identifier (left/right) or experiment identifier as fit for your setup.

```
bb_wdd --video_device ./Downloads/SomeVideoRecording.mp4 --cam_identifier cam0 --width 2048 --height 1080 --fps 60 --bee_length 44 --no_warmup --subsample 8
```

#### Specify sensitivity arguments

The default values for the parameters that control the sensitivity (mainly `binarization_threshold`) will most likely not work for your data leading to either too many or too few detections.

`binarization_threshold` controls the sensitivity for every pixel to the frequency of 12 to 15 Hz. Multiple such detections will then be merged and constitute one waggle run (or another signal). The value of this will depend on the video data. To get a feeling of the range of values, have a look at the output of the WDD, which might look like this:
`Currently processing with FPS: 63.3 | Max DD: 8.18 | [cam0             ./Downloads/omeVideoRecording.mp4]]`. The "Max DD" value is the same unit as the `binarization_threshold`.

`max_frame_distance` controls the allowed gaps in seconds between single responses of pixels above the `binarization_threshold` within a detection that are then merged into one waggle.
`min_num_detections` controls the minimum duration in seconds that one waggle run should have. In practice, this is the time between the first and last pixel response that are merged into one waggle (wherein each individual response has a gap to the last one of max `max_frame_distance`).

Let's add them to the first example:
```
bb_wdd --video_device ./Downloads/SomeVideoRecording.mp4 --cam_identifier cam0 --width 2048 --height 1080 --fps 60 --bee_length 44 --no_warmup --subsample 8 \
       --binarization_threshold 6 --max_frame_distance 0.4 --min_num_detections 0.25
```

### Automatic optimization of arguments

The WDD supports both evaluating a video and calculating some scores with given settings and also automatically adjusting the parameters to a video.
You will need to manually annotate the waggles in that video file, though. The WDD takes a CSV file with the annotations.
The file is passed either to the `eval` parameter to calculate scores after the video was processed or to the `autoopt` parameter to automatically search for the best parameters.

For the timestamps in the ground truth data to match the video, you will have to pass the `--start_timestamp` (ISO format) argument to the WDD.
In this example here, it might look like this `--start_timestamp 2020-08-30T10:00:00+00:00`.

Such a CSV file might look like this.
```
,start_frame,end_frame,origin_x,origin_y,end_x,end_y,start_ts,end_ts
0,589,629,906.0,762.0,934.0,763.0,2020-08-30 10:00:09.816667+00:00,2020-08-30 10:00:10.483333+00:00
1,784,827,898.0,745.0,933.0,751.0,2020-08-30 10:00:13.066667+00:00,2020-08-30 10:00:13.783333+00:00
2,953,1010,894.0,753.0,932.0,766.0,2020-08-30 10:00:15.883333+00:00,2020-08-30 10:00:16.833333+00:00
3,1131,1175,925.0,748.0,957.0,757.0,2020-08-30 10:00:18.850000+00:00,2020-08-30 10:00:19.583333+00:00
4,1306,1352,939.0,775.0,970.0,775.0,2020-08-30 10:00:21.766667+00:00,2020-08-30 10:00:22.533333+00:00
5,1592,1619,952.0,767.0,964.0,782.0,2020-08-30 10:00:26.533333+00:00,2020-08-30 10:00:26.983333+00:00
6,1788,1815,939.0,758.0,961.0,757.0,2020-08-30 10:00:29.800000+00:00,2020-08-30 10:00:30.250000+00:00
7,0,23,952.0,756.0,967.0,760.0,2020-08-30 10:00:00+00:00,2020-08-30 10:00:00.383333+00:00
```

The WDD can also read .pickle files containing pandas dataframes with the same columns or .annotations.json files as generated by the [BioTracker](https://github.com/BioroboticsLab/biotracker_core/).


### More arguments

There are more helpful arguments, e.g. to specify a ROI, to show a debug display, to record a video from the camera.
For the full list, see `bb_wdd --help`.

Deep Learning Decoding
----------------------

By default, the WDD uses computer vision based decoding of the waggles. The accuracy can be increased by installing the python package that contains the filter network
[bb_wdd_filter](https://github.com/BioroboticsLab/bb_wdd_filter/), downloading a trained model, and passing the path to the model with the `filter_model_path` argument.
This will also add the classification result and confidence to the metadata (possible classes are 'other', 'waggle', 'ventilating', 'activating' (where the latter is the shaking signal)).
