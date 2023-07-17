from collections import defaultdict
import datetime
import math
import numpy as np
import pytz
import glob
import hyperopt
from hyperopt import hp
from wdd.evaluation import load_ground_truth, calculate_precision_recall_f, calculate_scores, WaggleMetadataSaver
from wdd.main import run_wdd

def run_optimization(video_device, bee_length, gt_path, start_timestamp, fps, **kwargs):

    if start_timestamp is None and ".annotations.json" in gt_path:
        # Make sure we have a start timestamp for loading BioTracker GT data.
        start_timestamp = pytz.UTC.localize(datetime.datetime(1970, 1, 1, 0))
    elif isinstance(start_timestamp, str):
        start_timestamp = datetime.datetime.fromisoformat(start_timestamp).astimezone(pytz.utc)

    kwargs["fps"] = fps
    kwargs["no_fullframes"] = True
    kwargs["no_saving"] = True
    kwargs["stop_processing_on_low_fps"] = True

    all_video_files = glob.glob(video_device)
    print(all_video_files)
    all_ground_truth = [load_ground_truth(gt_path, video_path=video_path,
                            start_timestamp=start_timestamp, fps=fps) for video_path in all_video_files]


    search_space = dict(
        bee_length = hp.quniform("bee_length", bee_length * 0.8, bee_length * 1.2, q=1),
        subsample = hp.choice("subsample", list(np.arange(
                                                max(int(math.log(bee_length, 2) / math.log(5, 2)), 1),
                                                int(math.log(bee_length, 2))))),
        binarization_threshold = hp.uniform("binarization_threshold", 2, 10),
        max_frame_distance = hp.uniform("max_frame_distance", 0.25, 0.6),
        min_num_detections = hp.uniform("min_num_detections", 0.05, 0.3),
    )

    def objective(fun_kwargs):
        fun_kwargs = {**kwargs, **fun_kwargs}
        fun_kwargs["start_timestamp"] = start_timestamp.isoformat()
        fun_kwargs["no_warmup"] = True
        fun_kwargs["verbose"] = False
        fun_kwargs["quiet"] = True
        fun_kwargs["bee_length"] = int(fun_kwargs["bee_length"])
        fun_kwargs["subsample"] = int(fun_kwargs["subsample"])


        all_results = defaultdict(list)
        for (video_file, ground_truth) in zip(all_video_files, all_ground_truth):
            fun_kwargs["video_device"] = video_file
            saver = WaggleMetadataSaver()
            fun_kwargs["export_steps"] = [saver]
            processing_fps = run_wdd(**fun_kwargs)
            results = calculate_scores(saver.all_waggles, ground_truth, bee_length=bee_length, verbose=False)
            
            results["fps"] = processing_fps

            for key, val in results.items():
                all_results[key].append(val)
            
            # No need to pursue configurations that are too slow anyway.
            if processing_fps <= fps / 2:
                return dict(status=hyperopt.STATUS_FAIL)

        all_count_names = ("true_positives", "positives", "predicted_positive", "false_positives")
        other_keys = set(all_results.keys()) - set(all_count_names)
        for count_name in all_count_names:
            all_results[count_name] = np.nansum(all_results[count_name])
        all_results = {**all_results, **calculate_precision_recall_f(all_results["positives"], all_results["predicted_positive"], all_results["true_positives"])}

        for key in other_keys:
            all_results[key] = np.nanmean(all_results[key])

        all_results["loss"] = 1.0 - all_results["f_0.5"]

        all_results["status"] = hyperopt.STATUS_OK
        return all_results

    trials = hyperopt.Trials()
    for i in range(100):
        best = hyperopt.fmin(objective, search_space, algo=hyperopt.tpe.suggest, max_evals=(i + 1) * 20, show_progressbar=True, trials=trials)
        print("Epoch {} finished, current best parameters and score:".format(i + 1))
        print(hyperopt.space_eval(search_space, best))
        print(trials.best_trial["result"])
    