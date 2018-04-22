# Dependencies:
# !pip install numpy
# !pip install imageio
# !pip install matplotlib

import loaders
import transforms
import writers
import pprint
pp = pprint.PrettyPrinter(depth=6)
import sys
import json

# Settings
#data_path  = ".\\..\\Recorder\\FeatureSetBuilder\\Experiments\\experiment4.config"

class Procesor():
    def __init__(self, config):
        # Load the config
        with open(config) as json_data:
            config = json.load(json_data)

        self.loader = None
        self.writer = None
        self.transforms = []


        # Load the loader
        assert "loader" in config, "No loader found."
        if "random" in config["loader"]:
            print("Creating loader: random_loader")
            self.loader = loaders.LoaderRandom(config["loader"]["random"])
        else:
            print("ERROR: Loader not found")

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
                else:
                    print("ERROR: Transform not found '%s'" % transform["name"])

        # Load the writer
        assert "writer" in config, "No writer found."
        if "random_chunk_writer" in config["writer"]:
            self.writer = writers.random_chunk_writer(config["writer"]["random_chunk_writer"])

        assert self.loader is not None, "No loader created."
        assert self.writer is not None, "No writer created."



if( sys.argv[1] == "process" ):
	print("Processing experiment config frames from path %s." % (sys.argv[2]))
	exp = Processor(sys.argv[2])
	#exp.process()
else:
	print("ERROR: Invalid command %s. Must be play or process." % sys.argv[1])


