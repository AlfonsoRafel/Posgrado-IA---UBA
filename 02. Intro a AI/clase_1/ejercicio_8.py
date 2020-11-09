import numpy as np
import pickle


class Ratings(object):
    instance = None

    def __new__(cls, file):
        if Ratings.instance is None:
            print("__new__ object created")
            Ratings.instance = super(Ratings, cls).__new__(cls)
            return Ratings.instance
        else:
            return Ratings.instance

    def __init__(self, file):
        print("__init__")
        self.file = file
        try:
            raw_data = pickle.load(open(self.file + '.pkl', "rb"))
            print('Pickle file Opened')
        except:
            # Create structured numpy array
            structure = [('userId', np.int32),('movieId', np.int32),('rating', np.float32),('timestamp', np.int64)]
            structure = np.dtype(structure)
            # Read csv and save it as structured array
            raw_data = np.genfromtxt(self.file + '.csv', delimiter=',', skip_header=True, dtype=structure)
            # Save as pickle format
            pickle.dump(raw_data, open(self.file + '.pkl', 'wb'))
            print('Pickle file created')