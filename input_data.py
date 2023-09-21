# Define the quarter car model parameters
from roadprofile_management.loaders import SpeedDescriber


m_s = 1400/2  # Sprung mass (kg)
k_s = 34000  # Spring stiffness (N/m)
c_s = 3000  # Damping coefficient (Ns/m)
k_t = 384000  # Tire stiffness (N/m)
c_t = 0  # Tire damping (Ns/m)
m_u = 80  # Unsprung mass

roaddata_folder = r'C:\TRABAJO\CONICET\datasets\A 3D road surface topography\RoadData'
speed_descr = SpeedDescriber(t_accel=20, t_const=50, t_decel=10.1, max_speed=20)
file_qty = 295
delta_t = 0.001

# roaddata_folder = r'C:\TRABAJO\CONICET\datasets\SimpleSyntheticRoadData'
# speed_descr = SpeedDescriber(t_accel=1, t_const=9, t_decel=1, max_speed=1)
# file_qty = 1
# delta_t = 0.001
