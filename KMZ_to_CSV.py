# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import numpy as np
from zipfile import ZipFile
import xml.sax
import xml.sax.handler
from csv import reader
import math
from math import sin, cos, sqrt, atan2, radians
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from os import path
import collections
import sys
filename = 'MapTest.kmz'

kmz = ZipFile(filename, 'r')
kml = kmz.open('doc.kml', 'r')

#def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
 #   print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
  #  orb = cv2.ORB_create()

# Press the green button in the gutter to run the script.
#if __name__ == '__main__':
 #   print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/



class PlacemarkHandler(xml.sax.handler.ContentHandler):
    def __init__(self):
        self.inName = False  # handle XML parser events
        self.inPlacemark = False
        self.mapping = {}
        self.buffer = ""
        self.name_tag = ""

    def startElement(self, name, attributes):
        if name == "Placemark":  # on start Placemark tag
            self.inPlacemark = True
            self.buffer = ""
        if self.inPlacemark:
            if name == "name":  # on start title tag
                self.inName = True  # save name text to follow

    def characters(self, data):
        if self.inPlacemark:  # on text within tag
            self.buffer += data  # save text if in title

    def endElement(self, name):
        self.buffer = self.buffer.strip('\n\t')

        if name == "Placemark":
            self.inPlacemark = False
            self.name_tag = ""  # clear current name

        elif name == "name" and self.inPlacemark:
            self.inName = False  # on end title tag
            self.name_tag = self.buffer.strip()
            self.mapping[self.name_tag] = {}
        elif self.inPlacemark:
            if name in self.mapping[self.name_tag]:
                self.mapping[self.name_tag][name] += self.buffer
            else:
                self.mapping[self.name_tag][name] = self.buffer
        self.buffer = ""


parser = xml.sax.make_parser()
handler = PlacemarkHandler()
parser.setContentHandler(handler)
parser.parse(kml)
kmz.close()


def build_table(mapping):
    sep = ','

    output = ''
    points = ''
    lines = ''
    shapes = ''
    for key in mapping:
        coord_str = mapping[key]['coordinates'] + sep

        if 'LookAt' in mapping[key]:  # points
            points +=   coord_str + "\n"
        elif 'LineString' in mapping[key]:  # lines
            lines +=  coord_str + "\n"
        else:  # shapes
            shapes += coord_str + "\n"
    output += points + lines + shapes
    return output


outstr = build_table(handler.mapping)
out_filename = filename[:-3] + "csv" #output filename same as input plus .csv
f = open(out_filename, "w")
f.write(outstr)
f.close()
print(outstr)





