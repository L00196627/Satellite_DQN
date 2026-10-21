"""
    Definition of the Satellite class.

    This module defines the `Satellite` class used to represent satellites
    in the satellite network simulation. It includes attributes for the
    satellite's TLE data, distance from the observer, access point status.
"""

from skyfield.api import EarthSatellite

class Satellite:

    def __init__(self, satellite : EarthSatellite, distance, tle, is_acc_point):
        self.satellite = satellite
        self.distance = distance            # distance from the OBSERVER
        self.tle = tle                      
        self.is_acc_point = is_acc_point

        self.neighbors = {}                 # Dictionary of His Neighbors
        
        self.OGMs_History = []          # List of GMOs Reviewed in the Past
        self.OGMs = []                  # List of GMOs to be processed
        self.OGMs_NP = []               # List of Received GMOs (Not Processed)
        self.ogm_table = {}             # {'originator': [ 'neighbor': 'count']
        self.ogm_sequence = 0           # Count of GMOs Issued
