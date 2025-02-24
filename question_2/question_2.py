#!/usr/bin/env python

from os.path import isdir
from os import mkdir

from numpy import array, meshgrid, arange, mean
from matplotlib import pyplot as plt

data_directory = "Local_density_of_states_near_band_edge"
if not isdir(data_directory):
	print("Downloading data...")
	from urllib.request import urlretrieve
	mkdir(data_directory)
	for level in range(11):
		urlretrieve(f"https://raw.githubusercontent.com/Physics-129AL/Local_density_of_states_near_band_edge/refs/heads/main/local_density_of_states_for_level_{level}.txt", f"{data_directory}/{level}.txt")

def load_text(filename):
	with open(filename) as file:
		return array([list(map(float, line.split(", "))) for line in file])


# c
def extract_subregion(data_matrix):
	return data_matrix[180:220, 100:130]

average_values = [mean(extract_subregion(load_text(f"{data_directory}/{level}.txt"))) for level in range(11)]
plt.plot(average_values)
plt.xlabel("Level")
plt.ylabel("Average Local Density of States in a Subregion")
plt.show()