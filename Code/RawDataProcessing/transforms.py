import cv2

class Transform():
    def __init__(self):
        pass
    
    def requires_prerun(self):
        return False 

    def prerun(self, data):
        pass
    
    def process(self, data):
        pass


class OneHot(Transform):
    def __init__(self, config):
        self.input = config["input"]
        self.output = config["output"]
        self.dictionary = []

    def requires_prerun(self):
        return True

    def prerun(self, data):
        # Gets run with all the data prior to real run. Used to build term dictionary
        if data[self.input] not in self.dictionary:
            # New dictionary term
            self.dictionary.insert(data[self.input])

    def process(self, data):
        # TODO, 2d ARRAY SUPPORT: If input is a 2d array, we treat each input as one-hot separately with it's own dictionary

        # Build the one-hot array
        result = []
        for value in data[self.input]:
            result.append( self.dictionary.index(value) )

        # Write the result
        data[self.output] = result


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


class NormalizePerChannel(Transform):
    def __init__(self, config):
        self.input = config["input"]
        self.output = config["output"]
        self.min = config["min"]
        self.max = config["max"]

    def process(self, data):
        pass