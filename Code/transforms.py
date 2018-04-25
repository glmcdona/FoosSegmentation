import cv2
import pprint
import numpy as np
pp = pprint.PrettyPrinter(depth=6)
import os
import random
import math
from chunk import *

class Transform():
    def __init__(self):
        pass

    def prerun(self, data):
        pass
    
    def process(self, data):
        pass
    
    def finalize(self):
        pass

    def stop_all_on_return_null(self):
        return False

    def length(self):
        return None

class VideoLoaderRandom(Transform):
    def __init__(self, config):
        # Special frame loader that loads a randomly sample frame from any position in any video
        # until there are no frames left.

        self.output = config["output"]

        # Create the chunks
        self.chunks = []
        print("Adding folder '%s'" % config["folder"])
        for root, dirs, files in os.walk(config["folder"]):
            for file in files:
                #print("File '%s'" % file)            
                if file.endswith(config["endswith"]):
                    #print("Adding chunk from file '%s'" % file)
                    self.chunks.append( Chunk(root, file, config["value_loader"], config["repeats"]) )
        
        # Build the set chunk+frame pairs as the load order
        self.load_sequence = []
        self.distribution = {}
        for i in range(0,len(self.chunks)):
            group = ''.join(i for i in self.chunks[i].video_filename if not i.isdigit())
            if group not in self.distribution:
                self.distribution[group] = self.chunks[i].length
            else:
                self.distribution[group] += self.chunks[i].length
            
            for j in range(0, self.chunks[i].length):
                self.load_sequence.append([i, j])
        
        if "take_first_x_percent" in config:
            self.load_sequence = self.load_sequence[0:math.floor(len(self.load_sequence) * (config["take_first_x_percent"] / 100.0))]
        
        if "take_last_x_percent" in config:
            self.load_sequence = self.load_sequence[math.ceil(len(self.load_sequence) * (1.0 - config["take_last_x_percent"] / 100.0)):-1]

        # Shuffle the frame order
        random.shuffle(self.load_sequence)
        self.frame_index = 0
        print("Loaded %i frames in random loader." % len(self.load_sequence))
        print("Distribution:")
        print(self.distribution)

    def stop_all_on_return_null(self):
        return True

    def length(self):
        return len(self.load_sequence)
    
    def process(self, data):
        # Get the next frame along with the data
        if self.frame_index >= len(self.load_sequence):
            return None
        
        chunk_index, index = self.load_sequence[self.frame_index]
        self.frame_index += 1

        # Get the frame and vlues
        frame, values = self.chunks[chunk_index].get_frame(index)
        data[self.output] = frame

        # Merge the values
        for name, value in values.items():
            data[name] = value
        
        return data
    
    def finalize(self):
        self.frame_index = 0


class BasicWriter(Transform):
    def __init__(self, config):
        self.folder = config["folder"]
        self.filename_prefix = config["filename_prefix"]
        self.frames_per_chunk = config["frames_per_chunk"]
        self.frame_to_output = config["frame_to_output"]
        self.values_to_output = config["values_to_output"]
        self.frames_written = 0
        self.current_chunk = 0

        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        self.video_writer = None
        self.json_output_data = {}

    def process(self, data):
        # Get current chunk
        chunk_number = int( self.frames_written / self.frames_per_chunk )

        # Create the video writer
        if self.video_writer is None or chunk_number != self.current_chunk:
            if self.video_writer is not None:
                self.finalize_chunk()
            
            self.current_chunk = chunk_number

            # Load the frame size
            shape = data[self.frame_to_output].shape

            # Create the writer
            fourcc = cv2.VideoWriter_fourcc(*'XVID') #cv2.VideoWriter_fourcc(*'DIVX')
            self.video_writer = cv2.VideoWriter( os.path.join(self.folder, "%s_%i.avi" % (self.filename_prefix, self.current_chunk)), fourcc, 30.0, (shape[1], shape[0]), True )
            

        # Write the camera frame
        self.video_writer.write(data[self.frame_to_output])

        # Add the output data
        for key, value in data.items():
            if key in self.values_to_output:
                if key not in self.json_output_data:
                    self.json_output_data[key] = []
                if type(value) is np.ndarray:
                    self.json_output_data[key].append(value.tolist())
                else:
                    self.json_output_data[key].append(value)
        
        self.frames_written += 1
        return data
    
    def finalize(self):
        self.finalize_chunk()
        self.current_chunk = 0
        self.frames_written = 0
        self.video_writer = None

    def finalize_chunk(self):
        # Release the video writer
        self.video_writer.release()
        
        # Write the json data
        with codecs.open(os.path.join(self.folder, "%s_%i.json" % (self.filename_prefix, self.current_chunk)), 'w', encoding='utf-8') as outfile:
            json.dump(self.json_output_data, outfile, separators=(',', ':'), sort_keys=True, indent=4)
            self.json_output_data = {}
        
        pass

class Require(Transform):
    def __init__(self, config):
        self.input = config["input"]

    def process(self, data):
        if self.input not in data:
            return None

        if data[self.input] is None:
            return None

        return data

class OneHot(Transform):
    def __init__(self, config):
        self.input = config["input"]
        self.output = config["output"]
        self.dictionary = config["dictionary"]

    def prerun(self, data):
        # Gets run with all the data prior to real run. Used to build term dictionary
        if data[self.input] not in self.dictionary:
            # New dictionary term
            self.dictionary.insert(data[self.input])

    def process(self, data):
        # TODO, 2d ARRAY SUPPORT: If input is a 2d array, we treat each input as one-hot separately with it's own dictionary

        # Build the one-hot array
        result = np.zeros((len(self.dictionary), len(data[self.input])))
        for i in range(len(data[self.input])):
            result[self.dictionary.index(data[self.input][i]),i] = 1

        # Write the result
        data[self.output] = result

        return data


class Resize(Transform):
    def __init__(self, config):
        self.input = config["input"]
        self.output = config["output"]
        self.size = config["size"]
        self.resize_mode = config["resize_mode"]

    def process(self, data):
        # Resize
        result = cv2.resize(data[self.input], (self.size[1], self.size[0]), interpolation=cv2.INTER_AREA)
        
        # Write the result
        data[self.output] = result

        return data


class NormalizePerChannel(Transform):
    def __init__(self, config):
        self.input = config["input"]
        self.output = config["output"]
        self.min = config["min"]
        self.max = config["max"]

    def process(self, data):
        #pp.pprint(data)
        #pp.pprint(data["frame"].shape)
        #norm_image = np.array(data["frame"], dtype=np.float32)
        #pp.pprint(norm_image.shape)
        #norm_image = np.ascontiguousarray(norm_image)
        #pp.pprint(norm_image.shape)
        new_frame = data[self.input].astype(np.float32)
        for channel in range(data[self.input].shape[2]):
            min = np.min(new_frame[:,:,channel])
            max = np.max(new_frame[:,:,channel])
            
            new_frame[:,:,channel] = ( (new_frame[:,:,channel] - min) / (max - min) ) * (self.max - self.min) + self.min

        data[self.output] = new_frame
        return data
