## Random Forests for Bees

Using county industry changes to predict honey bee populations.

- [Our Input CoLab (Preprocessing)](https://colab.research.google.com/drive/1a8lbM7ceGGnaDe0kc1X0QqrZELsJINpb?usp=sharing)
- [Our Output CoLab](https://colab.research.google.com/drive/1y2A_XOFQrfu0HfXDPt2erg43Kn7Tc7xz?usp=sharing)
- ["Run Models" with Bees](../industries/)

- [Colab for creating 2-columns targets](https://colab.research.google.com/drive/11R3nSxPn91yTUBWhANBgdKCX0-YV1Dtk#scrollTo=Y9Un4FVwnxth)

NOTE: bees-targets.csv was a copy of bees-targets-increase2022.csv
bees-targets-top-20-percent.csv is the top 20% of colony density (not change over time). It's a copy of bees-population-usda.csv
bees-targets-top-20-percent.csv is the default when the target path is simply "bees"

Backup and run locally in [models/location-forest](../../models/location-forest/):

	python location-forest-input-bkup.ipynb bees
	python location-forest-output-bkup.ipynb bees


[2-column Target tables](https://github.com/ModelEarth/RealityStream/tree/main/input/bees/targets) containing county Fips.


[Our Run Models colab](../industries/) merges 2-column bee targets data for counties with features data with rows for industries and demographic data for each county.


### Bee Pollinators

<div style="overflow:auto; margin-top:0px; padding-right:50px">

  <div style="font-size:16px">
  <b><span class="yeartext"></span>[Prior change] predicting [future] change at locations or in industry mix</b><br>
  For model training, a "y" column value of 1 indicate locations where [Attribute(s)] that changed in a [prior year] predict a later year.<br><br>
  </div>

  <div style="background:#fff; padding:20px; max-width:600px">
	  <img src="https://model.earth/community-forecasting/about/img/random-forest.webp" style="width:100%;"><br>
	</div>

  <div style="display:none;font-size:12pt;line-height:16pt;padding-top:20px">
    Best Params: 
    max depth: 8; <!-- max number of levels in each decision tree -->
    n-estimators: 100 <!-- number of trees in the foreset --><br>

    Accuracy before tuning: 69%.&nbsp;
    Accuracy after tuning: 71%.
  </div>
  
</div>

[Prior Bees Output](../../output/bees/)


# Bee Population Density Analysis

## Overview
This readme file contains a [**Google Colab notebook**](https://colab.research.google.com/drive/1X04_N4E-WpcNRrolB2VgnvMfa8It99ZW?usp=sharing) that processes and analyzes honeybee population data. The dataset comes from the [**USDA National Agricultural Statistics Service (NASS) Quick Stats**](https://quickstats.nass.usda.gov/), and geographic data is retrieved from the [**U.S. Census Bureau's GEOINFO API**](https://api.census.gov/data/2023/geoinfo?get=GEO_ID,NAME,AREALAND,AREAWATR&for=county:*).

The goal is to calculate county-level bee population density and classify the top 20% as high-density (1). The notebook generates a csv file , containing FIPS codes and their binary classification (1 for high bee population density and 0 otherwise). 

The **bee population dataset is stored as a CSV file** located at: bees/inputs/target/bees-targets-top-20-percent.csv

This dataset was sourced directly from the [**USDA Quick Stats**](https://quickstats.nass.usda.gov/) website and contains **county-level bee population data** for different years(2002,2007,2012,2017,2022) across all states in USA.

### **Processing Steps**
1. **Data Cleaning**:
   - Entries with missing values or **"(D)"** (undisclosed data) are removed.
   - The **county names** are converted to lowercase for consistency.

2. **Fetching County-Level Land Area Data**:
   - The **U.S. Census Bureau GEOINFO API** is used to retrieve **land area data for all U.S. counties**.
   - County names and state names are extracted and formatted.
   - The **total land area (in km²)** is computed for each county.

3. **Merging Bee Population Data with Geographic Data**:
   - The **bee population dataset (2017 data) is read from**:
     ```
     bees/inputs/target/bees-targets-top-20-percent.csv
     ```
   - This dataset is filtered and merged with the **county land area dataset** using county names as the key.
   - FIPS (Federal Information Processing Standards) codes are retained for location identification.

4. **Bee Population Density Calculation**:
   - Bee population density is computed as:
     ```python
     bee_population_density = bee_population / total_area_km²
     ```
   - This provides a **standardized metric** to compare bee populations across counties.

5. **Creating a Binary Classification Target**:
   - The top **20% of counties with the highest bee population density** are labeled as `1` (high-density), while others are labeled as `0`.
   - This classification can be useful for **predictive modeling** in environmental studies.

## How to Use
1. **Open the Notebook** in Google Colab:
   - [BeeDensityTopCounties Colab Link](https://colab.research.google.com/drive/1X04_N4E-WpcNRrolB2VgnvMfa8It99ZW?usp=sharing)
2. **Run All Code Cells**:
   - The script will automatically load and process the data.
3. **Modify the bee-data**:
   - Adjust parameters in the bee-data dataframe to fetch data for different years or regions.


