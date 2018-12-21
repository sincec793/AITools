import numpy as np
import pandas as pd
import random
import cv2 as cv
class Dataset:
    #Class to handle various types of data. Currently supports rectangular data

    #Initialize dataset
    #num_sets: number of distinct parts in the data i.e (Numeric data, Text, Labels)
    def __init__(self, num_sets):
        self.numsets = num_sets
        self.data = {}

    #Load a CSV dataset
    def load_csv(self, file, dataName):
        df = pd.read_csv(file)
        self.data[dataName] = np.array(df)

    #Takes a list of strings that are filepaths to the images
    def load_images(self, img_list, colName='Images'):
        imgs = np.array([cv.imread(file) for file in img_list])
        self.data[colName] = imgs

    def get_data(self, dataName):
        return self.data[dataName]

    def sample(self, n=10):
        indicies = random.sample((0, min([len(v) for k,v in self.data])), n)
        return tuple([v[indicies] for k,v in self.data])

