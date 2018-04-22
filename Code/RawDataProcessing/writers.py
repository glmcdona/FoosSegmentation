
import os
import cv2
import json
import numpy as np
import codecs

class BasicWriter():
    def __init__(self, config):
        self.folder = config["folder"]
        self.filename_video = config["video"]
        self.filename_output = config["output"]
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        if os.path.isfile(os.path.join(self.folder, self.filename_video)):
            os.remove(os.path.join(self.folder, self.filename_video))

        if os.path.isfile(os.path.join(self.folder, self.filename_output)):
            os.remove(os.path.join(self.folder, self.filename_output))

        self.video_writer = None
        self.json_output_data = {}

    def add(self, data):
        # Create the video writer
        if self.video_writer is None:
            # Load the frame size
            shape = data["frame"].shape

            # Create the writer
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            self.video_writer = cv2.VideoWriter( os.path.join(self.folder, self.filename_video), fourcc, 30.0, shape[0:2])

        # Write the camera frame
        self.video_writer.write(data["frame"])

        # Add the output data
        for key, value in data.items():
            if key is not "frame":
                if key not in self.json_output_data:
                    self.json_output_data[key] = []
                if type(value) is np.ndarray:
                    self.json_output_data[key].append(value.tolist())
                else:
                    self.json_output_data[key].append(value)
        
        pass
    
    def finalize(self):
        # Release the video writer
        self.video_writer.release()
        
        # Write the json data
        with codecs.open(os.path.join(self.folder, self.filename_output), 'w', encoding='utf-8') as outfile:
            json.dump(self.json_output_data, outfile, separators=(',', ':'), sort_keys=True, indent=4)
        
        
        pass