# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline

# Importing the dataset
dataset = pd.read_csv('4.csv')

# Slicing
dataset = dataset[['ID', 'Community Area', 'Primary Type', 'Latitude', 'Longitude']].loc[dataset['Primary Type']
 .isin(['THEFT', 'BATTERY', 'CRIMINAL DAMAGE', 'NARCOTICS', 'ASSAULT', 'OTHER OFFENSE', 'BURGLARY'
        , 'DECEPTIVE PRACTICE', 'MOTOR VEHICLE THEFT', 'ROBBERY'])]
dataset = dataset.reset_index(drop=True)

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder = OneHotEncoder()
label_encoder = LabelEncoder()
data_label_encoded = label_encoder.fit_transform(dataset['Primary Type'])
dataset['Primary Type'] = data_label_encoded
data_feature_one_hot_encoded = encoder.fit_transform(dataset[['Primary Type']].as_matrix())

# Scatter-plot
lng = dataset.Longitude
lat = dataset.Latitude

# Mercator projection
def lat_lng_to_pixels(lat, lng):
    lat_rad = lat * np.pi / 180.0
    lat_rad = np.log(np.tan((lat_rad + np.pi / 2.0) / 2.0))
    x = 100 * (lng + 180.0) / 360.0
    y = 100 * (lat_rad - np.pi) / (2.0 * np.pi)
    return (x, y)

px, py = lat_lng_to_pixels(lat, lng)

plt.figure(figsize=(8, 6))
plt.scatter(px, py, s=.1, alpha=.03)
plt.axis('equal')
plt.xlim(25.50, 25.70)
plt.ylim(-37.25, -37.10)
plt.axis('off')


## misc

import matplotlib.colors as colors
import matplotlib.cm as cmx

uniq = list(set(dataset['Primary Type']))

# Set the color map to match the number of species
z = range(1,len(uniq))
plasma = plt.get_cmap('seismic')
cNorm  = colors.Normalize(vmin=0, vmax=len(uniq))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plasma)

# Plot each species
plt.figure(figsize=(80, 60))
for i in range(len(uniq)):
    indx = dataset['Primary Type'] == uniq[i]
    plt.scatter(px[indx], py[indx], s=.3, color=scalarMap.to_rgba(i), label=uniq[i], alpha=1)
plt.axis('equal')
plt.xlim(25.50, 25.70)
plt.ylim(-37.25, -37.10)
plt.axis('off')

#### arson

arson_set = dataset.loc[dataset['Primary Type'] == "ARSON"]
arr_lng = arson_set.Longitude
arr_lat = arson_set.Latitude
arrpx, arrpy = lat_lng_to_pixels(arr_lat, arr_lng)
plt.figure(figsize=(8, 6))
plt.scatter(arrpx, arrpy, s=1, alpha=1)
plt.axis('equal')
plt.xlim(25.50, 25.70)
plt.ylim(-37.25, -37.10)
plt.axis('off')


#### narcotics

nar_set = dataset.loc[dataset['Primary Type'] == "NARCOTICS"]
nar_lng = nar_set.Longitude
nar_lat = nar_set.Latitude
narpx, narpy = lat_lng_to_pixels(nar_lat, nar_lng)
plt.figure(figsize=(8, 6))
plt.scatter(narpx, narpy, s=0.01, alpha=0.1)
plt.axis('equal')
plt.xlim(25.50, 25.70)
plt.ylim(-37.25, -37.10)
plt.axis('off')

#### theft

thf_set = dataset.loc[dataset['Primary Type'] == "THEFT"]
thf_lng = thf_set.Longitude
thf_lat = thf_set.Latitude
thfpx, thfpy = lat_lng_to_pixels(thf_lat, thf_lng)
plt.figure(figsize=(8, 6))
plt.scatter(thfpx, thfpy, s=0.001, alpha=0.1)
plt.axis('equal')
plt.xlim(25.50, 25.70)
plt.ylim(-37.25, -37.10)
plt.axis('off')

#### homicide

hom_set = dataset.loc[dataset['Primary Type'] == "HOMICIDE"]
hom_lng = hom_set.Longitude
hom_lat = hom_set.Latitude
hompx, hompy = lat_lng_to_pixels(hom_lat, hom_lng)
plt.figure(figsize=(8, 6))
plt.scatter(hompx, hompy, s=0.1, alpha=0.4)
plt.axis('equal')
plt.xlim(25.50, 25.70)
plt.ylim(-37.25, -37.10)
plt.axis('off')

#### offense involving children

chd_set = dataset.loc[dataset['Primary Type'] == "OFFENSE INVOLVING CHILDREN"]
chd_lng = chd_set.Longitude
chd_lat = chd_set.Latitude
chdpx, chdpy = lat_lng_to_pixels(chd_lat, chd_lng)
plt.figure(figsize=(8, 6))
plt.scatter(chdpx, chdpy, s=0.1, alpha=0.4)
plt.axis('equal')
plt.xlim(25.50, 25.70)
plt.ylim(-37.25, -37.10)
plt.axis('off')

#### battery

bat_set = dataset.loc[dataset['Primary Type'] == "BATTERY"]
bat_lng = bat_set.Longitude
bat_lat = bat_set.Latitude
batpx, batpy = lat_lng_to_pixels(bat_lat, bat_lng)
plt.figure(figsize=(8, 6))
plt.scatter(batpx, batpy, s=0.1, alpha=0.4)
plt.axis('equal')
plt.xlim(25.50, 25.70)
plt.ylim(-37.25, -37.10)
plt.axis('off')

#### criminal damage

cdm_set = dataset.loc[dataset['Primary Type'] == "CRIMINAL DAMAGE"]
cdm_lng = cdm_set.Longitude
cdm_lat = cdm_set.Latitude
cdmpx, cdmpy = lat_lng_to_pixels(cdm_lat, cdm_lng)
plt.figure(figsize=(8, 6))
plt.scatter(cdmpx, cdmpy, s=0.1, alpha=0.4)
plt.axis('equal')
plt.xlim(25.50, 25.70)
plt.ylim(-37.25, -37.10)
plt.axis('off')

#### weapons violation

wev_set = dataset.loc[dataset['Primary Type'] == "WEAPONS VIOLATION"]
wev_lng = wev_set.Longitude
wev_lat = wev_set.Latitude
wevpx, wevpy = lat_lng_to_pixels(wev_lat, wev_lng)
plt.figure(figsize=(8, 6))
plt.scatter(wevpx, wevpy, s=0.1, alpha=0.4)
plt.axis('equal')
plt.xlim(25.50, 25.70)
plt.ylim(-37.25, -37.10)
plt.axis('off')