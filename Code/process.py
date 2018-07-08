# Dependencies:
# !pip install numpy
# !pip install imageio
# !pip install matplotlib

import transforms
import pprint
pp = pprint.PrettyPrinter(depth=6)
import sys
import json
import keras

# Settings
#data_path  = ".\\..\\Recorder\\FeatureSetBuilder\\Experiments\\experiment4.config"

class Processor():
    def __init__(self, config, subscription_key=""):
        keras.backend.clear_session()
        sys.setrecursionlimit(100000)

        # Load the config
        with open(config) as json_data:
            config = json.load(json_data)

        self.transforms = []
        
        # Load the data transforms
        if "transforms" in config:
            for transform in config["transforms"]:
                if "name" in transform:
                    print("Creating transform: %s" % transform["name"])
                    if transform["name"] == "run_if":
                        # Add all the if-statement transforms
                        pass
                    elif transform["name"] == "one-hot":
                        self.transforms.append( transforms.OneHot(transform) )
                    elif transform["name"] == "resize":
                        self.transforms.append( transforms.Resize(transform) )
                    elif transform["name"] == "resize_to_other":
                        self.transforms.append( transforms.ResizeToOther(transform) )
                    elif transform["name"] == "add_random_number":
                        self.transforms.append( transforms.AddRandomNumber(transform))
                    elif transform["name"] == "normalize_channels":
                        self.transforms.append( transforms.NormalizePerChannel(transform) )
                    elif transform["name"] == "random_video_loader":
                        self.transforms.append( transforms.VideoLoaderRandom(transform) )
                    elif transform["name"] == "single_frame_loader_middle":
                        self.transforms.append( transforms.SingleFrameLoaderMiddle(transform) )
                    elif transform["name"] == "basic_writer":
                        self.transforms.append( transforms.BasicWriter(transform) )
                    elif transform["name"] == "basic_writer_named":
                        self.transforms.append( transforms.BasicWriterNamed(transform) )
                    elif transform["name"] == "json_writer":
                        self.transforms.append( transforms.JsonWriter(transform) )
                    elif transform["name"] == "require":
                        self.transforms.append( transforms.Require(transform) )
                    elif transform["name"] == "replace":
                        self.transforms.append( transforms.Replace(transform) )
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
                    elif transform["name"] == "line_selection":
                        self.transforms.append( transforms.LineSelection(transform) )
                    elif transform["name"] == "point_selection":
                        self.transforms.append( transforms.PointSelection(transform) )
                    elif transform["name"] == "print_distribution":
                        self.transforms.append( transforms.PrintDistribution(transform) )
                    elif transform["name"] == "frame_difference":
                        self.transforms.append( transforms.FrameDifference(transform) )
                    elif transform["name"] == "threshold":
                        self.transforms.append( transforms.Threshold(transform) )
                    elif transform["name"] == "channel_max":
                        self.transforms.append( transforms.ChannelMax(transform) )
                    elif transform["name"] == "polygon_selection":
                        self.transforms.append( transforms.PolygonSelection(transform) )
                    elif transform["name"] == "draw_lines":
                        self.transforms.append( transforms.DrawLines(transform) )
                    elif transform["name"] == "draw_polygon":
                        self.transforms.append( transforms.DrawPolygon(transform) )
                    elif transform["name"] == "draw_contours":
                        self.transforms.append( transforms.DrawContours(transform) )
                    elif transform["name"] == "zeros_like":
                        self.transforms.append( transforms.ZerosLike(transform) )
                    elif transform["name"] == "concat_frames":
                        self.transforms.append( transforms.ConcatFrames(transform) )
                    elif transform["name"] == "to_int_frame":
                        self.transforms.append( transforms.ToIntFrame(transform) )
                    elif transform["name"] == "contours_find":
                        self.transforms.append( transforms.ContoursFind(transform) )
                    elif transform["name"] == "contours_adjust":
                        self.transforms.append( transforms.ContoursAdjust(transform) )
                    elif transform["name"] == "contours_to_frames":
                        self.transforms.append( transforms.ContoursToFrames(transform) )
                    elif transform["name"] == "polygon_to_contours":
                        self.transforms.append( transforms.PolygonToContours(transform) )
                    elif transform["name"] == "first_contour_to_polygon":
                        self.transforms.append( transforms.FirstContourToPolygon(transform) )
                    elif transform["name"] == "select_random":
                        self.transforms.append( transforms.SelectRandom(transform) )
                    elif transform["name"] == "bing_image_loader":
                        self.transforms.append( transforms.BingImageLoader(transform, subscription_key) )
                    elif transform["name"] == "merge_two_frames_by_polygon":
                        self.transforms.append( transforms.MergeTwoFramesByPolygon(transform) )
                    else:
                        print("ERROR: Transform not found '%s'" % transform["name"])
                elif "_comment_" in transform:
                    print("%s" % transform["_comment_"])
                else:
                    print("ERROR: Empty transform found!")

        
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
        subscription_key = ""
        if len(sys.argv) > 3:
            subscription_key = sys.argv[3]
        exp = Processor(sys.argv[2], subscription_key)
        exp.process_all()
    else:
        print("ERROR: Invalid command %s. Must be play or process." % sys.argv[1])


