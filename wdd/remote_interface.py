import datetime
import multiprocessing.connection
import numpy as np
import queue
import threading
import time

class ResultsSender():

    def __init__(self, address_string):
        
        address, port, authkey = address_string.split(":")
        self.address = address
        self.port = int(port)
        self.authkey = authkey

        self.con = None
        self.queue = queue.Queue()
        self.running = True

        self.thread = threading.Thread(target=self.send_results, args=())
        self.thread.daemon = False
        self.thread.start()

    def close(self):
        self.running = False
        self.queue.put(None)
        self.thread.join()

    def send_results(self):

        try:
            while self.running:
                if self.con is None:
                    try:
                        self.con = multiprocessing.connection.Client((self.address, self.port), authkey=self.authkey.encode())
                        print("\nConnected external interface.")
                    except ConnectionRefusedError as e:
                        time.sleep(2.0)
                        continue
                
                msg = self.queue.get()
                if msg is None:
                    continue

                try:
                    self.con.send(msg)
                except ConnectionResetError as e:
                    print("\nExternal connection reset.")
                    self.con = None
                    time.sleep(0.5)

        finally:
            if self.con is not None:
                self.con.send("close")

    def __call__(self, waggle, full_frame_rois, metadata_dict, **kwargs):
        
        payload = dict(
            timestamp_waggle=waggle.timestamp,
            system_timestamp_waggle=waggle.system_timestamp,
            system_timestamp_sending=datetime.datetime.utcnow(),
            x=np.median(metadata_dict["x_coordinates"]),
            y=np.median(metadata_dict["y_coordinates"]),
            cam_id=metadata_dict["cam_id"]
        )

        if "waggle_angle" in metadata_dict:
            payload["waggle_angle"] = metadata_dict["waggle_angle"]
            payload["waggle_duration"] = metadata_dict["waggle_duration"]

        self.queue.put(payload)

        return waggle, full_frame_rois, metadata_dict, kwargs
