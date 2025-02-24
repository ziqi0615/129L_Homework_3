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

# a
heatmap_directory = "local_density_of_states_heatmap"
if not isdir(heatmap_directory):
	print("Generating heatmaps...")
	mkdir(heatmap_directory)
	for level in range(11):
		data_matrix = load_text(f"{data_directory}/{level}.txt")
		plt.imshow(data_matrix, cmap="hot")
		plt.colorbar()
		plt.title(f"Local Density of States for Level {level}")
		plt.savefig(f"{heatmap_directory}/{level}.png")
		plt.close()

# b
heightmap_directory = "local_density_of_states_height"
if not isdir(heightmap_directory):
	print("Generating height plots...")
	mkdir(heightmap_directory)
	for level in range(11):
		data_matrix = load_text(f"{data_directory}/{level}.txt")
		x_values, y_values = meshgrid(arange(data_matrix.shape[1]), arange(data_matrix.shape[0]))
		fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
		ax.plot_surface(x_values, y_values, data_matrix, cmap="hot")
		ax.set_title(f"Local Density of States for Level {level}")
		plt.savefig(f"{heightmap_directory}/{level}.png")
		plt.close()

# c
def extract_subregion(data_matrix):
	return data_matrix[180:220, 100:130]

average_values = [mean(extract_subregion(load_text(f"{data_directory}/{level}.txt"))) for level in range(11)]
plt.plot(average_values)
plt.xlabel("Level")
plt.ylabel("Average Local Density of States in a Subregion")
plt.show()
