# Define the quarter car model parameters
from roadprofile_management.loaders import SpeedDescriber


m_s = 1400/2  # Sprung mass (kg)
k_s = 34000  # Spring stiffness (N/m)
c_s = 3000  # Damping coefficient (Ns/m)
k_t = 384000  # Tire stiffness (N/m)
c_t = 0  # Tire damping (Ns/m)
m_u = 80  # Unsprung mass

roaddata_folder = r'C:\TRABAJO\CONICET\datasets\A 3D road surface topography\RoadData'
file_ini = 0
file_qty = 295
delta_t = 0.001
timesteps_skip = 10


speed_describers = [None,
                    SpeedDescriber(t_accel=20, t_const=100, t_decel=10.1, max_speed=10),
                    SpeedDescriber(t_accel=20, t_const=150, t_decel=10.1, max_speed=5)]
generate_dataset = True
