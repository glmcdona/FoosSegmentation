# Dependencies:
# !pip install numpy
# !pip install imageio
# !pip install matplotlib

import transforms
import pprint
pp = pprint.PrettyPrinter(depth=6)
import sys
import json

# Settings
#data_path  = ".\\..\\Recorder\\FeatureSetBuilder\\Experiments\\experiment4.config"

class Processor():
    def __init__(self, config):
        # Load the config
        with open(config) as json_data:
            config = json.load(json_data)

        self.transforms = []
        
        # Load the data transforms
        if "transforms" in config:
            for transform in config["transforms"]:
                print("Creating transform: %s" % transform["name"])
                if transform["name"] == "one-hot":
                    self.transforms.append( transforms.OneHot(transform) )
                elif transform["name"] == "resize":
                    self.transforms.append( transforms.Resize(transform) )
                elif transform["name"] == "normalize_channels":
                    self.transforms.append( transforms.NormalizePerChannel(transform) )
                elif transform["name"] == "random_video_loader":
                    self.transforms.append( transforms.VideoLoaderRandom(transform) )
                elif transform["name"] == "basic_writer":
                    self.transforms.append( transforms.BasicWriter(transform) )
                elif transform["name"] == "require":
                    self.transforms.append( transforms.Require(transform) )
                elif transform["name"] == "randomize_frame":
                    self.transforms.append( transforms.RandomizeFrame(transform) )
                elif transform["name"] == "single_video_loader":
                    self.transforms.append( transforms.SingleVideoLoader(transform) )
                elif transform["name"] == "run_model":
                    self.transforms.append( transforms.RunModel(transform) )
                elif transform["name"] == "onehot_to_name":
                    self.transforms.append( transforms.OneHotToName(transform) )
                elif transform["name"] == "frame_write_text":
                    self.transforms.append( transforms.FrameWriteText(transform) )
                elif transform["name"] == "show_frame":
                    self.transforms.append( transforms.ShowFrame(transform) )
                elif transform["name"] == "print_distribution":
                    self.transforms.append( transforms.PrintDistribution(transform) )
                else:
                    print("ERROR: Transform not found '%s'" % transform["name"])
        
        # Load the length
        self.length = self.transforms[0].length()
        self.current_frame = 0
        

    def process_all(self):
        # Loop through the pipeline until the loader is done
        count = 0
        while self.get_next_frame() is not None:
            if count % 100 == 0:
                print("Frame %i of %i" % (count, self.length))
            count += 1
    
    def get_next_frame(self):
        # Process this frame through the transforms
        done = False
        data = {}
        for transform in self.transforms:
            data = transform.process(data)
            if data is None:
                #print("Transform of type '%s' returned None." % transform.__class__.__name__)
                if transform.stop_all_on_return_null():
                    done = True
                break

        if done:
            # Finalize transforms
            for transform in self.transforms:
                transform.finalize()

            # Restart
            self.current_frame = 0

            # Return Null as the end value
            return None

        self.current_frame += 1

        # Skip blank results
        if data is None:
            return self.get_next_frame()
        
        # Return the resulting frame
        return data


if __name__ == "__main__":
    if( sys.argv[1] == "process" ):
        print("Processing experiment config frames from path %s." % (sys.argv[2]))
        exp = Processor(sys.argv[2])
        exp.process_all()
    else:
        print("ERROR: Invalid command %s. Must be play or process." % sys.argv[1])


