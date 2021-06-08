import click
import datetime
import os
import sys
import time
import math
import numpy as np

from wdd.main import run_wdd

def show_default_option(*args, **kwargs):
    return click.option(show_default=True, *args, **kwargs)


@click.command()
@show_default_option(
    "--capture_type",
    default="PyCapture2",
    help="Whether to use OpenCV or PyCapture2 to aquire images",
)
@click.option(
    "--video_device", required=True, help="OpenCV video device. Can be camera index or video path"
)
@show_default_option("--height", default=180, help="Video frame height in px (before subsampling).")
@show_default_option("--width", default=342, help="Video frame width in px (before subsampling).")
@click.option("--subsample", default=0, help="Fast subsampling by using every Xth pixel of the images.")
@show_default_option("--fps", default=60, help="Frames per second")
@show_default_option("--bee_length", default=7, help="Approximate length of a bee in px (before subsampling).")
@show_default_option(
    "--binarization_threshold",
    default=6.0,
    help="Binarization threshold for waggle detection in log scale. Can be used to tune sensitivity/specitivity",
)
@show_default_option(
    "--max_frame_distance",
    default=0.2,
    help="Maximum time inbetween frequency detections within one waggle in seconds",
)
@show_default_option(
    "--min_num_detections", default=0.1, help="Minimum time of a waggle in seconds"
)
@click.option(
    "--output_path", type=click.Path(exists=True), required=True, help="Output path for results."
)
@click.option(
    "--cam_identifier", required=True, help="Identifier of camera (used in output storage path)."
)
@show_default_option("--debug", is_flag=True, help="Enable debug outputs/visualization")
@show_default_option(
    "--debug_frames",
    default=11,
    help="Only visualize every debug_frames frame in debug mode (can be slow if low)",
)
@click.option(
    "--no_multiprocessing",
    is_flag=True,
    help="Do not use a multiprocessing queue to fetch the images."
)
@click.option(
    "--no_warmup",
    is_flag=True,
    help="Do not warm up the image retrieval.."
)
@click.option(
    "--start_timestamp",
    default=None,
    help="Instead of using the wall-clock, generate camera timestamps based on the FPS starting at this iso-formatted timestamp (example: '2019-08-30T12:30:05.000100+00:00')."
)
@click.option(
    "--autoopt",
    default="",
    help="Automatically optimize hyperparameters given a CSV file with annotations."
)
@click.option(
    "--eval",
    default="",
    help="Check the detected waggles against a given CSV file and print evaluation results."
)
@click.option(
    "--roi",
    nargs=4,
    type=int,
    default=None,
    help="Specify a region of interest in pixels in the original video. The four arguments are 'left, top, width, height'."
)
def main(
    bee_length,
    autoopt,
    **kwargs
):
    if not autoopt:
        run_wdd(bee_length=bee_length, **kwargs)
        print("\nStopping.")
    else:
        print("Optimizing hyperparameters..")
        import hyperopt
        from hyperopt import hp
        from wdd.evaluation import load_ground_truth, calculate_scores, WaggleMetadataSaver

        ground_truth = load_ground_truth(autoopt)
        
        search_space = dict(
            bee_length = hp.quniform("bee_length", bee_length * 0.8, bee_length * 1.2, q=1),
            subsample = hp.choice("subsample", list(np.arange(
                                                    max(int(math.log(bee_length, 2) / math.log(5, 2)), 1),
                                                    int(math.log(bee_length, 2))))),
            binarization_threshold = hp.uniform("binarization_threshold", 2, 10),
            max_frame_distance = hp.uniform("max_frame_distance", 0.1, 0.5),
            min_num_detections = hp.uniform("min_num_detections", 0.05, 0.3),
        )

        def objective(fun_kwargs):
            fun_kwargs = {**kwargs, **fun_kwargs}
            fun_kwargs["no_warmup"] = True
            fun_kwargs["verbose"] = False
            fun_kwargs["bee_length"] = int(fun_kwargs["bee_length"])
            fun_kwargs["subsample"] = int(fun_kwargs["subsample"])

            
            saver = WaggleMetadataSaver()
            fun_kwargs["export_steps"] = [saver]
            fps = run_wdd(**fun_kwargs)

            results = calculate_scores(saver.all_waggles, ground_truth, bee_length=bee_length, verbose=False)
            results["fps"] = fps
            results["loss"] = 1.0 - results["f_0.5"]
            results["status"] = hyperopt.STATUS_OK
            return results

        trials = hyperopt.Trials()
        best = hyperopt.fmin(objective, search_space, algo=hyperopt.tpe.suggest, max_evals=20, show_progressbar=True, trials=trials)

        print("Optimization finished!")
        print("Best parameters:")
        print(hyperopt.space_eval(search_space, best))
        print(trials.best_trial["result"])



if __name__ == "__main__":
    main()
