import os
import cv2
import numpy as np
import types
import json
import codecs
import threading
import pprint as pp

class Chunk():
    def __init__(self, video_root, video_filename, value_loader_config, repeats):
        self.file = os.path.join(video_root, video_filename)
        self.repeats = repeats
        self.video_filename = video_filename
        self.lock = threading.Lock()

        # Add the value loader
        self.value_loader = None
        if value_loader_config is not None:
            if "delimited_filename" in value_loader_config:
                self.value_loader = ValueLoader_DelimitedFilename(video_filename, value_loader_config["delimited_filename"])
            elif "json" in value_loader_config:
                self.value_loader = ValueLoader_Json(video_root, video_filename, value_loader_config["json"])

        # Calculate the chunk length
        self.reader = cv2.VideoCapture(self.file)
        self.num_frames = int(self.reader.get(cv2.CAP_PROP_FRAME_COUNT))
        print("'%s': %i frames found." % (self.file,self.num_frames))

        # Calculate if this chunk should be repeated at all
        repeat_times = 1
        if self.repeats is not None:
            for repeat in self.repeats:
                if repeat["match_type"] == "partial_match_filename":
                    if repeat["value_match"] in video_filename:
                        repeat_times = repeat_times * repeat["repeat_times"]
        if repeat_times != 1:
            print("'%s': Repeating %f times." % (self.file,repeat_times))
        
        self.length = int(self.num_frames * repeat_times)

        self.width = None
        self.height = None
        if self.reader.isOpened() and self.num_frames > 0:
            self.reader.set(1, 0)
            ret, first_frame = self.reader.read()
            if ret == True:
                self.width = np.shape(first_frame)[1]
                self.height = np.shape(first_frame)[0]
    
    def get_frame_value(self, index):
        # Load the value for the frame
        if self.value_loader is None:
            return None
        
        return self.value_loader.get_values(index)

    def get_frame(self, index):
        frame = None
        values = {}

        # Load the camera frame
        with self.lock:
            if self.reader.isOpened():
                self.reader.set(1, (index % self.num_frames))
                ret, new_frame = self.reader.read()
                if ret == True and new_frame is not None:
                    frame = new_frame

        # Load and merge the associated values
        if self.value_loader is None:
            values = None
        else:
            values = self.value_loader.get_values(index)

        return frame, values

class ValueLoader_DelimitedFilename():
    def __init__(self, filename, value_loader_config):
        # Parse the filename according to the config as the value
        tokens = filename.split(".")[0].split( value_loader_config["delimiter"] )
        self.values = {}
        if "filename" in value_loader_config:
            self.values[value_loader_config["filename"]] = filename
        
        for name, indices in value_loader_config["token_load"].items():
            if type(indices) is not list:
                indices = [indices]
            
            items = []
            for index in indices:
                items.append( tokens[index] )
            
            # Add this named value
            self.values[name] = items

        self.filename_noext = filename.split(".")[0]
        self.values["filename_noext"] = self.filename_noext

    def get_values(self, index):
        # Constant set of values depending on filename
        return self.values

class ValueLoader_Json():
    def __init__(self, video_root, filename, value_loader_config):
        # Parse the filename according to the config as the value
        self.filename_noext = filename.split(".")[0]
        self.constant = False
        if "constant" in value_loader_config:
            self.constant = value_loader_config["constant"]

        json_file = os.path.join(video_root, self.filename_noext + ".json")
        if os.path.isfile(json_file):
            data = codecs.open(json_file, 'r', encoding='utf-8').read()
            self.data = json.loads(data)
            #pp.pprint(len(self.data["ball"]))
        else:
            self.data = {}

    def get_values(self, index):
        if self.constant:
            # Constant set of values for this loader
            values = {}
            for name, array in self.data.items():
                if len(array) > 0:
                    values[name] = array[0]
                else:
                    values[name] = []
        else:
            # Decode per the value for this index
            values = {}
            for name, array in self.data.items():
                values[name] = array[index]
        
        values["filename_noext"] = self.filename_noext
        return values