import datetime
import json
import numpy as np
import os
import pandas
import pytz

class WaggleMetadataSaver():
    def __init__(self):
        self.all_waggles = []
    def __call__(self, waggle, rois, metadata, **kwargs):
        self.all_waggles.append(metadata)
        return waggle, rois, metadata, kwargs

def load_ground_truth(path, start_timestamp=None, fps=None, video_path=""):

    if not os.path.exists(path):
        path = video_path + path
        if not os.path.exists(path):
            raise ValueError("Path for ground truth data file is invalid.")

    if path.endswith("pickle"):
        annotations = pandas.read_pickle(path)
    elif path.endswith("csv"):
        assert path.endswith("csv")
        annotations = pandas.read_csv(path, parse_dates=["start_ts", "end_ts"])
    elif path.endswith(".annotations.json"): # BioTracker annotations.
        with open(path, "r") as f:
            annotations = json.load(f)
            annotations = [a for a in annotations if a["type"] == "arrow"]

            def map_offset(frame_index):
                return start_timestamp + datetime.timedelta(seconds=frame_index / fps)
            annotations = pandas.DataFrame(annotations)
            annotations["start_ts"] = annotations.start_frame.apply(map_offset)
            annotations["end_ts"] = annotations.end_frame.apply(map_offset)
            for col in ("origin_x", "origin_y", "end_x", "end_y"):
                annotations[col] = annotations[col].astype(np.float32)
    else:
        raise ValueError("Unknown file format for annotations.")

    return annotations

def parse_waggle_metadata(waggle_metadata):

    waggle_metadata["timestamp_begin"] = datetime.datetime.fromisoformat(waggle_metadata["timestamp_begin"]).astimezone(pytz.utc)
    waggle_metadata["frame_timestamps"] = [datetime.datetime.fromisoformat(ts).astimezone(pytz.utc) for ts in  waggle_metadata["frame_timestamps"]]
    waggle_metadata["camera_timestamps"] = [datetime.datetime.fromisoformat(ts).astimezone(pytz.utc) for ts in  waggle_metadata["camera_timestamps"]]
                                    
    return waggle_metadata

def calculate_precision_recall_f(P, PP, TP):
    precision = TP/PP if PP != 0 else 0
    recall = TP/P

    results = dict(
        precision=precision,
        recall=recall,
    )

    for f in (0.5, 1, 2):
        f_score = 0.0
        if (precision + recall) > 0.0:
            f_score = (1.0 + f ** 2.0) * (precision * recall) / ((f ** 2.0 * precision) + recall)
        results["f_{}".format(f)] = f_score

    return results

def calculate_scores(all_waggle_metadata, ground_truth_df, bee_length, verbose=True):

    all_waggle_metadata = [parse_waggle_metadata(m) for m in all_waggle_metadata]

    waggles_df = []
    for row in all_waggle_metadata:
        x = np.median(row["x_coordinates"])
        y = np.median(row["y_coordinates"])
        begin = row["camera_timestamps"][0]
        end = row["camera_timestamps"][-1]
        response = np.median(row["responses"])
        
        angle, duration = None, None
        if "waggle_angle" in row:
            angle, duration = row["waggle_angle"], row["waggle_duration"]

        waggles_df.append(dict(
            x=x, y=y,
            start=pandas.Timestamp(begin), end=pandas.Timestamp(end),
            length=end - begin,
            length_s=(end - begin).total_seconds(),
            response=response,
            angle=angle, duration=duration,
        ))

    waggles_df = pandas.DataFrame(waggles_df)
    decoding_results = []
    
    hits = []

    if waggles_df.shape[0] > 0:

        waggles_df["start"] = pandas.to_datetime(waggles_df.start)
        waggles_df["end"] = pandas.to_datetime(waggles_df.end)
        print(ground_truth_df.end_ts.max(), type(ground_truth_df.end_ts.max()))
        print(waggles_df.end.max(), type(waggles_df.end.max()))
        waggles_df = waggles_df[waggles_df.end < ground_truth_df.end_ts.max()]
        matched_waggles = np.zeros(shape=(waggles_df.shape[0],), dtype=np.bool)

        for x0, y0, x1, y1, dt_begin, dt_end in ground_truth_df[["origin_x", "origin_y",
                                                            "end_x", "end_y",
                                                            "start_ts", "end_ts"]].itertuples(index=False):
            gt_vector = np.array([x1, y1]) - np.array([x0, y0])
            gt_vector[1] *= -1.0
            gt_angle = np.arctan2(gt_vector[1], gt_vector[0])
            gt_vector /= np.linalg.norm(gt_vector)
            
            p = bee_length
            x0, x1 = min(x0, x1), max(x0, x1)
            y0, y1 = min(y0, y1), max(y0, y1)
            waggles = waggles_df[~matched_waggles]
            waggles = waggles[waggles.x.between(x0 - p, x1 + p).values & waggles.y.between(y0 - p, y1 + p).values]
            waggles = waggles[(~(waggles.start > dt_end).values & ~(waggles.end < dt_begin).values)]
            if waggles.shape[0] == 0:
                hits.append(0)
                continue
            elif waggles.shape[0] > 1:
                if verbose:
                    print("Found more than one waggle for GT.")
                normalized_durations = waggles.duration.copy()
                normalized_durations[pandas.isna(normalized_durations)] = 0.0
                waggles_midpoint = (waggles.start + pandas.to_timedelta(normalized_durations / 2.0, unit="s"))
                gt_midpoint = (dt_begin + (dt_end - dt_begin) / 2.0)
                #print("two waggles", gt_midpoint, waggles_midpoint, dt_begin, dt_end, normalized_durations)
                offset = (waggles_midpoint - gt_midpoint).apply(lambda d: d.total_seconds()).values
                best_match_idx = np.argmin(np.abs(offset))
                waggles = waggles.iloc[best_match_idx:(best_match_idx+1), :]

            assert waggles.shape[0] == 1
            gt_duration = (dt_end - dt_begin).total_seconds()
            

            for waggle_idx in range(waggles.shape[0]):
                waggle_duration = waggles.duration.iloc[waggle_idx]
                waggle_angle = waggles.angle.iloc[waggle_idx]

                if waggle_angle is None or np.isnan(waggle_angle):
                    continue

                waggle_vector = np.array([np.cos(waggle_angle), np.sin(waggle_angle)])
                angle_dot = np.dot(gt_vector, waggle_vector)
                angle_error = np.arccos(angle_dot)
            
                duration_error = np.nan
                if not pandas.isnull(waggle_duration):
                    duration_error = (waggle_duration - gt_duration) ** 2.0
                decoding_results.append(dict(
                    angular_error_rad=angle_error,
                    angular_error_deg=angle_error / np.pi * 180.0,
                    duration_error=duration_error
                ))

            matched_waggles[np.array(waggles.index, dtype=np.int)] = 1
            hits.append(1)
        
        waggles_df["TP"] = matched_waggles
    else:
        matched_waggles = np.array([], dtype=np.bool)
        
    hits = np.array(hits, dtype=np.float32)

    N = np.sum(~matched_waggles)
    P = ground_truth_df.shape[0]
    PP = waggles_df.shape[0]
    TP = np.sum(hits)
    FP = np.sum(~matched_waggles)

    results = dict(
        positives=P,
        predicted_positive=PP,
        false_positives=FP,
        true_positives=TP,
    )

    results = {**results, **calculate_precision_recall_f(P, PP, TP)}

    if decoding_results:
        decoding_results = pandas.DataFrame(decoding_results)
        results["angular_error_rad"] = np.nanmean(decoding_results.angular_error_rad.values)
        results["angular_error_deg"] = np.nanmean(decoding_results.angular_error_deg.values)
        results["angular_error_deg"] = np.nanmean(decoding_results.angular_error_deg.values)
        results["duration_error"] = np.nanmean(decoding_results.duration_error.values)
    return results
