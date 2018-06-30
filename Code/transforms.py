import cv2
import pprint
import numpy as np
pp = pprint.PrettyPrinter(depth=6)
import os
import random
import math
from chunk import *
from keras.preprocessing.image import *
import keras

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

        self.output = None
        self.num_frames = 0
        if "output" in config:
            self.output = config["output"]
            self.num_frames = 1

        self.outputs = None
        if "outputs" in config:
            self.outputs = config["outputs"]
            self.num_frames = len(self.outputs)
        
        if "repeats" not in config:
            config["repeats"] = None

        if "value_loader" not in config:
            config["value_loader"] = None

        self.random = True
        if "random" in config:
            self.random = config["random"]

        self.frame_skipfirst_per_video = 0
        if "frame_skipfirst_per_video" in config:
            self.frame_skipfirst_per_video = config["frame_skipfirst_per_video"]

        # Create the chunks
        self.chunks = []
        print("Adding folder '%s'" % config["folder"])
        for root, dirs, files in os.walk(config["folder"]):
            for file in files:
                #print("File '%s'" % file)            
                if file.endswith(config["endswith"]):
                    include = True
                    if "inclusions" in config:
                        include = False
                        for inclusion in config["inclusions"]:
                            if inclusion in file:
                                include = True
                    
                    if "exclusions" in config:
                        for exclusion in config["exclusions"]:
                            if exclusion in file:
                                include = False

                    if include:
                        #print("Adding chunk from file '%s'" % file)
                        self.chunks.append( Chunk(root, file, config["value_loader"], config["repeats"]) )
        
        # Load the frame limit per video
        self.frame_limit_per_video = 1000000000
        if "frame_limit_per_video" in config:
            self.frame_limit_per_video = config["frame_limit_per_video"]

        # Build the set chunk+frame pairs as the load order
        self.load_sequence = []
        self.distribution = {}
        for i in range(0,len(self.chunks)):
            group = ''.join(i for i in self.chunks[i].video_filename if not i.isdigit())
            if group not in self.distribution:
                self.distribution[group] = self.chunks[i].length
            else:
                self.distribution[group] += self.chunks[i].length
            
            for j in range(0 + self.frame_skipfirst_per_video, min([self.chunks[i].length - self.num_frames + 1, self.frame_limit_per_video + self.frame_skipfirst_per_video - self.num_frames + 1])):
                self.load_sequence.append([i, range(j, j+self.num_frames)])
        
        if "take_first_x_percent" in config:
            self.load_sequence = self.load_sequence[0:math.floor(len(self.load_sequence) * (config["take_first_x_percent"] / 100.0))]
        
        if "take_last_x_percent" in config:
            self.load_sequence = self.load_sequence[math.ceil(len(self.load_sequence) * (1.0 - config["take_last_x_percent"] / 100.0)):-1]

        # Shuffle the frame order
        if self.random:
            random.shuffle(self.load_sequence)
        
        self.frame_index = 0
        print("Loaded %i frames in loader." % len(self.load_sequence))
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
        
        chunk_index, indices = self.load_sequence[self.frame_index]
        self.frame_index += 1

        if len(indices) == 1:
            # Process single frame and values
            if self.output is None:
                values = self.chunks[chunk_index].get_frame_value(indices[0])
            else:
                frame, values = self.chunks[chunk_index].get_frame(indices[0])
                data[self.output] = frame

            # Merge the values
            if values is not None:
                for name, value in values.items():
                    data[name] = value
        else:
            # Process array of frames and values.
            # Adds prefix of "<output name>_" to data values to keep them separate.
            for idx, index in enumerate(indices):
                output_name = self.outputs[idx]

                # Get the frame and values
                frame, values = self.chunks[chunk_index].get_frame(index)
                data[output_name] = frame

                # Merge the values
                if values is not None:
                    for name, value in values.items():
                        data["%s_%s" % (output_name,name)] = value
        
        return data
    
    def finalize(self):
        self.frame_index = 0



class SingleFrameLoaderMiddle(Transform):
    def __init__(self, config):
        # Special frame loader that loads a randomly sample frame from any position in any video
        # until there are no frames left.

        self.output = None
        if "output" in config:
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
        for i in range(0,len(self.chunks)):
            self.load_sequence.append([i, int(self.chunks[i].length/2)])
        
        self.frame_index = 0
        print("Loaded %i frames in single frame loader." % len(self.load_sequence))

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

        # Get the frame and values
        if self.output is None:
            values = self.chunks[chunk_index].get_frame_value(index)
        else:
            frame, values = self.chunks[chunk_index].get_frame(index)
            data[self.output] = frame

        # Merge the values
        for name, value in values.items():
            data[name] = value
        
        return data
    
    def finalize(self):
        self.frame_index = 0


class JsonWriter(Transform):
    def __init__(self, config):
        self.folder = config["folder"]
        filename_prefix = None
        if filename_prefix in config:
            self.filename_prefix = config["filename_prefix"]
        self.filename_prefix_inputname = None
        if "filename_prefix_inputname" in config:
            self.filename_prefix_inputname = config["filename_prefix_inputname"]
        self.values_per_chunk = config["values_per_chunk"]
        self.values_to_output = config["values_to_output"]
        self.values_written = 0
        self.current_chunk = 0

        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        self.json_output_data = {}

    def process(self, data):
        # Get current chunk
        chunk_number = int( self.values_written / self.values_per_chunk )

        # Finalize chunks
        if chunk_number != self.current_chunk:
            self.finalize_chunk()

        # Set the filename ooutput for the json
        if self.filename_prefix_inputname is not None:
            self.filename_prefix = data[self.filename_prefix_inputname]

        # Add the output data
        for key, value in data.items():
            if key in self.values_to_output:
                if key not in self.json_output_data:
                    self.json_output_data[key] = []
                if type(value) is np.ndarray:
                    self.json_output_data[key].append(value.tolist())
                else:
                    self.json_output_data[key].append(value)
        
        self.values_written += 1
        return data
    
    def finalize(self):
        self.finalize_chunk()
        self.current_chunk = 0
        self.values_written = 0

    def finalize_chunk(self):
        # Write the json data
        if len(self.json_output_data) > 0:
            with codecs.open(os.path.join(self.folder, "%s_%i.json" % (self.filename_prefix, self.current_chunk)), 'w', encoding='utf-8') as outfile:
                json.dump(self.json_output_data, outfile, separators=(',', ':'), sort_keys=True, indent=4)
                self.json_output_data = {}
        
        pass

class BasicWriter(Transform):
    def __init__(self, config):
        self.folder = config["folder"]
        self.filename_prefix = config["filename_prefix"]
        self.frames_per_chunk = config["frames_per_chunk"]
        self.frame_to_output = config["frame_to_output"]

        self.values_to_output = []
        if "values_to_output" in config:
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
        if len(self.json_output_data) > 0:
            with codecs.open(os.path.join(self.folder, "%s_%i.json" % (self.filename_prefix, self.current_chunk)), 'w', encoding='utf-8') as outfile:
                json.dump(self.json_output_data, outfile, separators=(',', ':'), sort_keys=True, indent=4)
                self.json_output_data = {}
        
        pass


class BasicWriterNamed(Transform):
    def __init__(self, config):
        self.folder = config["folder"]
        self.p_filename = config["p_filename"]
        self.frame_to_output = config["frame_to_output"]

        self.values_to_output = []
        if "values_to_output" in config:
            self.values_to_output = config["values_to_output"]

        self.frames_written = 0

        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        self.current_filename = None
        self.video_writer = None
        self.json_output_data = {}

    def process(self, data):
        # Create the video writer
        if self.video_writer is None or self.current_filename != data[self.p_filename]:
            if self.video_writer is not None:
                self.finalize_chunk()
            
            self.current_filename = data[self.p_filename]

            # Load the frame size
            shape = data[self.frame_to_output].shape

            # Create the writer
            fourcc = cv2.VideoWriter_fourcc(*'XVID') #cv2.VideoWriter_fourcc(*'DIVX')
            self.video_writer = cv2.VideoWriter( os.path.join(self.folder, self.current_filename.split(".")[0] + ".avi"), fourcc, 30.0, (shape[1], shape[0]), True )
            

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
        self.current_filename = None
        self.frames_written = 0
        self.video_writer = None

    def finalize_chunk(self):
        # Release the video writer
        self.video_writer.release()
        
        # Write the json data
        if len(self.json_output_data) > 0:
            with codecs.open(os.path.join(self.folder, "%s.json" % (self.current_filename.split(".")[0])), 'w', encoding='utf-8') as outfile:
                json.dump(self.json_output_data, outfile, separators=(',', ':'), sort_keys=True, indent=4)
                self.json_output_data = {}
        
        pass

class ZerosLike(Transform):
    def __init__(self, config):
        self.reference = config["reference"]
        self.output = config["output"]
    
    def process(self, data):
        data[self.output] = np.zeros_like(data[self.reference])
        return data

class Require(Transform):
    def __init__(self, config):
        self.input = config["input"]

        self.input_index = None
        if "input_index" in config:
            self.input_index = config["input_index"]
        
        self.condition = None
        if "condition" in config:
            self.condition = config["condition"]

        self.value = None
        if "value" in config:
            self.value = config["value"]

    def process(self, data):
        if self.condition is None or self.condition == "must exist":
            if self.input not in data:
                return None

            if data[self.input] is None:
                return None

            if self.input_index is not None:
                if data[self.input][self.input_index] is None:
                    return None

        elif self.condition == "does not exist":
            if self.input in data:
                if data[self.input] is not None:
                    if self.input_index is None:
                        return None
                    else:
                        if data[self.input][self.input_index] is not None:
                            return None
        elif self.condition == "equals":
            value = None
            if self.input in data:
                if data[self.input] is not None:
                    if self.input_index is not None:
                        value = data[self.input][self.input_index]
                    else:
                        value = data[self.input]

            if value != self.value:
                return None

        return data

class Replace(Transform):
    def __init__(self, config):
        self.input_output = config["input_output"]
        self.replacements = config["replacements"]


    def process(self, data):
        for replacement in self.replacements:
            if replacement["from"] == data[self.input_output]:
                data[self.input_output] = replacement["to"]

        return data

class AddRandomNumber(Transform):
    def __init__(self, config):
        self.output = config["output"]
        self.prng = np.random.RandomState()

    def process(self, data):
        # Add a random int
        data[self.output] = self.prng.randint(1,1000000)

        return data

class OneHot(Transform):
    def __init__(self, config):
        self.input = config["input"]
        self.output = config["output"]
        self.dictionary = config["dictionary"]
        self.out_of_map_value = None
        if "out_of_map_value" in config:
            self.out_of_map_value = config["out_of_map_value"]

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
            if data[self.input][i] in self.dictionary:
                result[self.dictionary.index(data[self.input][i]),i] = 1
            else:
                result = self.out_of_map_value

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


class FrameDifference(Transform):
    def __init__(self, config):
        self.inputs = config["inputs"]
        self.output = config["output"]
        self.abs = config["abs"]

    def process(self, data):
        # Frame difference
        if self.abs:
            result = abs(data[self.inputs[1]] - data[self.inputs[0]])
        else:
            result = data[self.inputs[1]] - data[self.inputs[0]]
        
        # Write the result
        data[self.output] = result

        return data

class Threshold(Transform):
    def __init__(self, config):
        self.input = config["input"]
        self.threshold = config["threshold"]
        self.min = config["min"]
        self.max = config["max"]

    def process(self, data):
        # Above threshold -> max
        # Below threshold -> min
        above = data[self.input] > self.threshold
        below = data[self.input] <= self.threshold

        data[self.input][above] = self.max
        data[self.input][below] = self.min
        return data

class ChannelMax(Transform):
    def __init__(self, config):
        self.input = config["input"]
        self.output = config["output"]

    def process(self, data):
        # Above threshold -> max
        # Below threshold -> min
        max_data = np.amax(data[self.input], 2)
        for i in range(3):
            data[self.output][:,:,i] = max_data
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
            if min != max:
                new_frame[:,:,channel] = ( (new_frame[:,:,channel] - min) / (max - min) ) * (self.max - self.min) + self.min
            else:
                new_frame[:,:,channel] = new_frame[:,:,channel] - min

        data[self.output] = new_frame
        return data

class RandomizeFrame(Transform):
    def __init__(self, config):
        self.input = config["input"]
        self.output = config["output"]

        self.p_seed = None
        if "p_seed" in config:
            self.p_seed = config["p_seed"]

        self.chance_no_change = 0.0
        if "chance_no_change" in config:
            self.chance_no_change = config["chance_no_change"]

        self.zoom_range = None
        if "zoom_range" in config:
            self.zoom_range = config["zoom_range"]
        
        self.rotation_range = None
        if "rotation_range" in config:
            self.rotation_range = config["rotation_range"]
        
        self.width_shift_range = None
        if "width_shift_range" in config:
            self.width_shift_range = config["width_shift_range"]
        
        self.height_shift_range = None
        if "height_shift_range" in config:
            self.height_shift_range = config["height_shift_range"]
        
        self.shear_range = None
        if "shear_range" in config:
            self.shear_range = config["shear_range"]
        
        self.fill_mode = "nearest"
        if "fill_mode" in config:
            self.fill_mode = config["fill_mode"]
        
        self.gaussian_noise = 0.0
        if "gaussian_noise" in config:
            self.gaussian_noise = config["gaussian_noise"]
        
        self.white_noise = "white_noise"
        if "white_noise" in config:
            self.white_noise = config["white_noise"]

        self.swap_channels = False
        if "swap_channels" in config:
            self.swap_channels = config["swap_channels"]

        self.invert_channels = False
        if "invert_channels" in config:
            self.invert_channels = config["invert_channels"]
        
        self.vertical_flip = False
        if "vertical_flip" in config:
            self.vertical_flip = config["vertical_flip"]
        
        self.horizontal_flip = None
        if "horizontal_flip" in config:
            self.horizontal_flip = config["horizontal_flip"]
        
        self.row_axis = 0
        if "row_axis" in config:
            self.row_axis = config["row_axis"]
        self.col_axis = 1
        if "col_axis" in config:
            self.col_axis = config["col_axis"]
        self.channel_axis = 2
        if "channel_axis" in config:
            self.channel_axis = config["channel_axis"]

        self.saturation_chance = 0.0
        if "saturation_chance" in config:
            self.saturation_chance = config["saturation_chance"]
        self.saturation_max_amount = 0.3
        if "saturation_max_amount" in config:
            self.saturation_max_amount = config["saturation_max_amount"]

        self.seed_random = 1
        if "seed" in config:
            self.seed_random = config["seed"]

        # Set the current random seed
        self.prng = np.random.RandomState(self.seed_random)

    def process(self, data):
        # Applies the configured random transformations to the input camera frame

        # Adapted based on Keras ImageDataGenerator, but with improvements
        #     https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py

        # Set the seed if it is loaded from a data value
        if self.p_seed is not None:
            self.prng = np.random.RandomState(data[self.p_seed])
            random.seed(data[self.p_seed])


        x = data[self.input]

        if not self.prng.uniform() < self.chance_no_change:
            # use composition of homographies
            # to generate final transform that needs to be applied
            if self.rotation_range:
                theta = np.pi / 180 * self.prng.uniform(-self.rotation_range, self.rotation_range)
            else:
                theta = 0

            if self.height_shift_range:
                tx = self.prng.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[self.row_axis]
            else:
                tx = 0

            if self.width_shift_range:
                ty = self.prng.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[self.col_axis]
            else:
                ty = 0

            if self.shear_range:
                shear = self.prng.uniform(-self.shear_range, self.shear_range)
            else:
                shear = 0

            if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
                zx, zy = 1, 1
            else:
                zx, zy = self.prng.uniform(self.zoom_range[0], self.zoom_range[1], 2)

            transform_matrix = None
            if theta != 0:
                rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                            [np.sin(theta), np.cos(theta), 0],
                                            [0, 0, 1]])
                transform_matrix = rotation_matrix

            if tx != 0 or ty != 0:
                shift_matrix = np.array([[1, 0, tx],
                                        [0, 1, ty],
                                        [0, 0, 1]])
                transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

            if shear != 0:
                shear_matrix = np.array([[1, -np.sin(shear), 0],
                                        [0, np.cos(shear), 0],
                                        [0, 0, 1]])
                transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

            if zx != 1 or zy != 1:
                zoom_matrix = np.array([[zx, 0, 0],
                                        [0, zy, 0],
                                        [0, 0, 1]])
                transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

            if transform_matrix is not None:
                h, w = x.shape[self.row_axis], x.shape[self.col_axis]
                transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
                
                x = apply_transform(x, transform_matrix, self.channel_axis,
                                    fill_mode=self.fill_mode, cval=0.0)

            if self.horizontal_flip:
                if self.prng.uniform() < 0.5:
                    x = flip_axis(x, self.col_axis)

            if self.vertical_flip:
                if self.prng.uniform() < 0.5:
                    x = flip_axis(x, self.row_axis)
            
            if self.swap_channels:
                # Randomly swap the channels
                # TODO: Make it generic using the column number members
                if self.prng.uniform() > 0.5:
                    order = list(range(3))
                    random.shuffle(order)
                    result = np.zeros_like(x)
                    for i in range(3):
                        result[:,:,i] = x[:,:,order[i]]
                    x = result

            if self.invert_channels:
                # Invert between 0 and 3 channels. 50% chance of no inverting
                if self.prng.uniform() > 0.5:
                    # Invert some channels
                    num_channels = self.prng.randint(1,3)
                    channels = list(range(3))
                    for i in range(num_channels):
                        random.shuffle(channels)
                        channel = channels.pop()
                        x[:,:,channel] = -x[:,:,channel] # A normalizer is assumed to restore it to the correct value range
            # Apply the noise
            max_color = np.max(x)
            min_color = np.min(x)
            value_range = max_color - min_color

            level = self.prng.uniform(0, self.white_noise * value_range)
            white_noise = self.prng.uniform(-level, level, x.shape)

            std_dev = self.prng.uniform(0, self.gaussian_noise * value_range)
            gaussian_noise = self.prng.normal(0, std_dev, x.shape)

            x = (x + white_noise + gaussian_noise)

            # Apply saturation
            if self.saturation_chance > 0 and self.prng.uniform() < self.saturation_chance:
                if self.prng.uniform() > 0.5:
                    # Saturate all the channels together
                    max_color = np.max(x)
                    min_color = np.min(x)
                    value_range = max_color - min_color
                    rand = self.prng.uniform()
                    if rand > 0.6:
                        # 40% chance Saturate brightest values of this channel
                        saturated_max = (max_color - value_range * ( self.saturation_max_amount * self.prng.uniform() ))
                        mask = x > saturated_max
                        x[mask] = saturated_max
                    elif rand > 0.2:
                        # 40% chance Saturate darkest values
                        saturated_min = (min_color + value_range * ( self.saturation_max_amount * self.prng.uniform() ))
                        mask = x < saturated_min
                        x[mask] = saturated_min
                    else:
                        # 20% chance Saturate both the brightest and darkest
                        saturated_max = (max_color - value_range * ( self.saturation_max_amount/2 * self.prng.uniform() ))
                        mask = x > saturated_max
                        x[mask] = saturated_max

                        saturated_min = (min_color + value_range * ( self.saturation_max_amount/2 * self.prng.uniform() ))
                        mask = x < saturated_min
                        x[mask] = saturated_min
                else:
                    # Saturate all or some channels differently
                    num_channels = self.prng.randint(1,3)
                
                    channels = list(range(3))
                    for i in range(num_channels):
                        random.shuffle(channels)
                        channel = channels.pop()
                        max_color = np.max(x[:,:,channel])
                        min_color = np.min(x[:,:,channel])
                        value_range = max_color - min_color
                        rand = self.prng.uniform()
                        if rand > 0.6:
                            # 40% chance Saturate brightest values of this channel
                            saturated_max = (max_color - value_range * ( self.saturation_max_amount * self.prng.uniform() ))
                            mask = x[:,:,channel] > saturated_max
                            x[mask,channel] = saturated_max
                        elif rand > 0.2:
                            # 40% chance Saturate darkest values
                            saturated_min = (min_color + value_range * ( self.saturation_max_amount * self.prng.uniform() ))
                            mask = x[:,:,channel] < saturated_min
                            x[mask,channel] = saturated_min
                        else:
                            # 20% chance Saturate both the brightest and darkest
                            saturated_max = (max_color - value_range * ( self.saturation_max_amount/2 * self.prng.uniform() ))
                            mask = x[:,:,channel] > saturated_max
                            x[mask,channel] = saturated_max

                            saturated_min = (min_color + value_range * ( self.saturation_max_amount/2 * self.prng.uniform() ))
                            mask = x[:,:,channel] < saturated_min
                            x[mask,channel] = saturated_min

        # Set the output
        data[self.output] = x

        return data


class SingleVideoLoader(Transform):
    def __init__(self, config):
        self.output = config["output"]
        self.video = None
        if "webcam_number" in config:
            self.video = cv2.VideoCapture(config["webcam_number"])
        if "video_file" in config:
            self.video = cv2.VideoCapture(config["video_file"])
        
        if "webcam_number" in config:
            self.num_frames = 1000000000000 # infinite
            print("Reading frames from webcam.")
        else:
            # Calculate the chunk length
            self.num_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
            print("%i frames found from video file." % (self.num_frames))

    def length(self):
        return self.num_frames

    def process(self, data):
        if(self.video.isOpened()):
            ret, frame = self.video.read()
            if ret == True:
                data[self.output] = frame
            else:
                return None
        else:
            return None

        # Continue
        return data
    
    def stop_all_on_return_null(self):
        return True

class FrameWriteText(Transform):
    def __init__(self, config):
        self.input_output = config["input_output"]
        self.line = config["line"]

        self.text_constant_prefix = ""
        if "text_constant_prefix" in config:
            self.text_constant_prefix = config["text_constant_prefix"]
        
        self.text_from_value = None
        if "text_from_value" in config:
            self.text_from_value = config["text_from_value"]

    def process(self, data):
        # Draw text on the frame
        text = ""
        if self.text_from_value is not None:
            text = self.text_constant_prefix + str(data[self.text_from_value])
        else:
            text = self.text_constant_prefix
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(data[self.input_output], text,(10,50+40*self.line), font, 1,(255,255,255),1,cv2.LINE_AA)
        
        # Continue
        return data

class ShowFrame(Transform):
    def __init__(self, config):
        self.input = config["input"]
        self.display = config["display"]
        cv2.namedWindow(self.display)

    def process(self, data):
        # Show the frame
        cv2.imshow(self.display,data[self.input])

        # Keystroke to quit or pause
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            data = None
        elif key == ord(' '):
            cv2.waitKey()
        
        # Continue
        return data
    
    def stop_all_on_return_null(self):
        return True

    def finalize(self):
        cv2.destroyWindow(self.display)


global refPt
refPt = (0,0)
def click_callback(event, x, y, flags, param):
	global refPt
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = (x, y)

class LineSelection(Transform):
    def __init__(self, config):
        self.input_frame = config["input_frame"]
        self.title = config["title"]
        self.line_names = config["line_names"]
        self.output_lines = config["output_lines"]

    def process(self, data):
        # Show the frame
        global refPt

        data[self.output_lines] = []
        for line in self.line_names:
            # Select this line
            cv2.namedWindow(line)
            cv2.setMouseCallback(line, click_callback)
            cv2.imshow(line, data[self.input_frame])

            # Select left point
            refPt = (0,0)
            key = 0
            while refPt == (0,0) and key != ord('q'):
                key = cv2.waitKey(1) & 0xFF
            
            if refPt == (0,0) or key == ord('q'):
                print("Cancelled selection of lines for this frame")
                cv2.destroyWindow(line)
                return None
            left = refPt

            # Select right point
            refPt = (0,0)
            key = 0
            while refPt == (0,0) and key != ord('q'):
                key = cv2.waitKey(1) & 0xFF
            
            if refPt == (0,0) or key == ord('q'):
                print("Cancelled selection of lines for this frame")
                cv2.destroyWindow(line)
                return None
            right = refPt
            
            # Swap left and right if needed
            if left[0] > right[0]:
                tmp = right
                right = left
                left = tmp

            # Add the two points that define the line
            data[self.output_lines].append([left, right])
            cv2.destroyWindow(line)
            cv2.line(data[self.input_frame], left, right, [0,255,0])
        
        # Continue
        return data


class PolygonSelection(Transform):
    def __init__(self, config):
        self.input_frame = config["input_frame"]
        self.title = config["title"]
        self.output = config["output"]
        self.num_vertices = config["num_vertices"]

    def process(self, data):
        # Show the frame
        global refPt

        data[self.output] = np.zeros((self.num_vertices, 2),dtype=int)
        for i in range(self.num_vertices):
            # Select this vertex
            cv2.namedWindow(self.title)
            cv2.setMouseCallback(self.title, click_callback)
            cv2.imshow(self.title, data[self.input_frame])

            # Select the point
            refPt = (0,0)
            key = 0
            while refPt == (0,0) and key != ord('q'):
                key = cv2.waitKey(1) & 0xFF
            
            if refPt == (0,0) or key == ord('q'):
                print("Cancelled selection of vertex")
                cv2.destroyWindow(self.title)
                return None
            vertex = refPt

            data[self.output][i,:] = vertex

            # Draw the point
            cv2.circle(data[self.input_frame], vertex, 3, (0,255,255))
            #cv2.fillConvexPoly(data[self.input_frame], data[self.output][0:i,:], (0,255,255))
        
        # Continue
        return data


class DrawLines(Transform):
    def __init__(self, config):
        self.input_frame = config["input_frame"]
        self.input_lines = config["input_lines"]
        self.width = config["width"]
        self.color = config["color"]

    def process(self, data):
        # Show the frame
        for line in data[self.input_lines]:
            cv2.line(data[self.input_frame], tuple(line[0]), tuple(line[1]), self.color, self.width)
        
        # Continue
        return data

class DrawPolygon(Transform):
    def __init__(self, config):
        self.input_frame = config["input_frame"]
        self.input_polygon = config["input_polygon"]
        self.fill = config["fill"]
        self.width = config["width"]
        self.color = config["color"]

    def process(self, data):
        # Show the frame
        if self.fill:
            cv2.fillConvexPoly(data[self.input_frame], np.array(data[self.input_polygon]), self.color)
        else:
            cv2.fillConvexPoly(data[self.input_frame], np.array(data[self.input_polygon]), self.color)
        
        # Continue
        return data

class RunModel(Transform):
    def __init__(self, config):
        self.input = config["input"]
        self.output = config["output"]
        self.model = keras.models.load_model(config["model_file"])

    def process(self, data):
        # Process the frame in the ML model
        result = self.model.predict( np.expand_dims(data[self.input], axis=0) )[0,:]
        data[self.output] = result
        
        # Continue
        return data

class OneHotToName(Transform):
    def __init__(self, config):
        self.input = config["input"]
        self.output = config["output"]
        self.map = config["map"]

        self.ignores = []
        if "ignores" in config:
            self.ignores = config["ignores"]

        self.output_prob = None
        if "output_prob" in config:
            self.output_prob = config["output_prob"]

    def process(self, data):
        for ignore in self.ignores:
            data[self.input][ignore] = 0

        # Convert a One-Hot scoring to the winning category and probability
        index = data[self.input].argmax()
        data[self.output] = self.map[index]

        if self.output_prob is not None:
            data[self.output_prob] = max(data[self.input])

        # Continue
        return data


class PrintDistribution(Transform):
    def __init__(self, config):
        self.input = config["input"]
        self.strip_numbers = False
        if "strip_numbers" in config:
            self.strip_numbers = config["strip_numbers"]
        self.counts = {}

    def process(self, data):
        term = str(data[self.input])
        if term is None:
            term = "None"
        if self.strip_numbers:
            term = ''.join(i for i in term if not i.isdigit())

        # Add the count
        if term not in self.counts:
            self.counts[term] = 0
        self.counts[term] += 1

        # Continue
        return data

    def finalize(self):
        # Print the distribution
        pp.pprint(self.counts)
