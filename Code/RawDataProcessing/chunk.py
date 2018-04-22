import os
import cv2
import numpy as np
import types

class Chunk():
    def __init__(self, video_root, video_filename, value_loader_config):
        self.file = os.path.join(video_root, video_filename)

        # Add the value loader
        self.value_loader = None
        if "delimited_filename" in value_loader_config:
            self.value_loader = value_loader_delimited_filename(video_filename, value_loader_config["delimited_filename"])

        # Calculate the chunk length
        self.reader = cv2.VideoCapture(self.file)
        self.num_frames = int(self.reader.get(cv2.CAP_PROP_FRAME_COUNT))
        print("'%s': %i frames found." % (self.file,self.num_frames))

        self.width = None
        self.height = None
        if self.reader.isOpened() and self.num_frames > 0:
            self.reader.set(1, 0)
            ret, first_frame = self.reader.read()
            if ret == True:
                self.width = np.shape(first_frame)[1]
                self.height = np.shape(first_frame)[0]

    def length(self):
        return self.num_frames
    
    def get_frame(self, index):
        data = {}

        # Load the camera frame
        if self.reader.isOpened():
            self.reader.set(1, index)
            ret, frame = self.reader.read()
            if ret == True and frame is not None:
                data["frame"] = frame

        # Load and merge the associated values
        values = self.value_loader.get_values(index)
        for name, value in values.items():
            data[name] = value

        return data

class ValueLoader_DelimitedFilename():
    def __init__(self, filename, value_loader_config):
        # Parse the filename according to the config as the value
        tokens = filename.split(".")[0].split( value_loader_config["delimiter"] )
        self.values = {}
        for name, indices in value_loader_config["token_load"].items():
            if type(indices) is not list:
                indices = [indices]
            
            items = []
            for index in indices:
                items.append( tokens[index] )
            
            # Add this named value
            self.values[name] = items

    def get_values(self, index):
        # Constant set of values depending on filename
        return self.values