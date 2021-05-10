import numpy as np

class PointGPS:
    def __init__(self,latitude,longitude,altitude):
        self.lat = latitude
        self.lon = longitude
        self.alt = altitude

class ParserGPS:
    """Convert GPS data to position [X,Y,Z] in meters""" 
    def __init__(self):
        """Class constructor"""
        self.pos = {
            'x' : [],
            'y' : [],
            'z' : []}
        self.R = 6371 

    def haversine_distance(self, point_1, point_2):
        """Calculate haversine distance. Source : https://en.wikipedia.org/wiki/Haversine_formula

        Args:
            point_1 (PointGPS): First point
            point_2 (PointGPS): Second point

        Returns:
            float: Calculated delta distance b/w two points in meters
        """
        r = self.R
        lat1 = point_1.lat
        lon1 = point_1.lon
        lat2 = point_2.lat
        lon2 = point_2.lon
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2-lat1)
        delta_lambda = np.radians(lon2-lon1)
        a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) *   np.sin(delta_lambda / 2)**2
        res = r * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
        return np.round(res, 2)

if __name__ == "__main__":
    p_NewYork = PointGPS(40.6976637,-74.1197643,None)
    p_denver = PointGPS(39.7645187,-104.9951948,None)
    p_miami = PointGPS(25.7825453,-80.2994985,None)
    p_chicago = PointGPS(41.8339037,-87.8720471,None)

    gps_parser = ParserGPS()

    dist_denver = gps_parser.haversine_distance(p_NewYork,p_denver)
    dist_miami = gps_parser.haversine_distance(p_NewYork,p_miami)
    dist_chicago = gps_parser.haversine_distance(p_NewYork,p_chicago)

    print("Distance to Denver from New York is  : {0}".format(dist_denver))
    print("Distance to Miami from New York is  : {0}".format(dist_miami))
    print("Distance to Chicago from New York is  : {0}".format(dist_chicago))