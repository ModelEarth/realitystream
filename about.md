## Run Models CoLab

- [Run Models (CoLab)](input/industries) - For features and targets merged on their location columns.
- [Models Overview](models)

In Run Models, the "features" dataset is merged with a 2-column "targets" dataset on-the-fly using either .csv files or Pandas to avoid storing merged .csv files. The location column joins features and targets.

We also support feature files that already contain a target column, like the [eye blink data Random Bits Forest (RBF)](models/random-bits-forest/). 

Location column data types:  
World Region (TBD), Country (2-char), State (2-char), County Fips (5-digits for state and county), Zip (5 char, 6 in China), or Brain Voxel (2 char)

The target does not need to a location ID. It can be an ID that clusters multiple location, as is the case for Brain Voxels that fire together when eye blinks occur. View [eye blink data .csv file](https://github.com/ModelEarth/realitystream/blob/main/models/random-bits-forest/blinks-input.csv) - each column is a voxel (location in brain).

## Default Data Sources

Our features-targets merge supports any data with a location column containing multiple locations.  
Our default data will always use County Fips so features and targets align.

**Industries (Features and Targets)** - County Fips
<a href="input/industries/">Industries Input Data</a>

**Bees (Target)** - County Fips
<a href="/bee-data/">Random Forest (Bees)</a>

**Trees (Target)** - County Fips
[Tree Targets](input/trees/)

**Blinks** - Rows are clusters of brain voxels - hence multiple locations have one target column
<a href="models/random-bits-forest/">Random Bits Forest (Blinks)</a><br>


You can add paths to external data by editing a copy of the [parameters.yaml](https://github.com/ModelEarth/realitystream/blob/main/parameters/parameters.yaml) file.


## Path Parameters

The term "features" is more prevalent in machine learning and data science.
"factors" has a stronger association with statistics and social sciences. The term factors is used for impact attributes like emissions.

TO DO: Add a python command that loads parameters.yaml to run [Run-Models-bkup.ipynb](https://github.com/ModelEarth/realitystream/tree/main/models) locally, so the user does not need to open a notebook. Pass a parameters.yaml path in. 

Parameters are loaded from the parameters.yaml file:

	python Run-Models-bkup.ipynb [raw path to parameters.yaml]

Example of parameters.yaml format:

	folder: naics6-bees-counties
	features: industries
		startyear: 2017
		endyear: 2021
	 	path: https://raw.githubusercontent.com/ModelEarth/community-timelines/main/training/naics{naics}/US/counties/{year}/US-{state}-training-naics{naics}-counties-{year}.csv
	targets: bees
		path: https://github.com/ModelEarth/bee-data/raw/main/targets/bees-targets-top-20-percent.csv
	models: lr, svc, rfc, rbf, xgboost

<!-- For later
	python Run-Models-bkup.ipynb [features] [target] [models]
-->

Each target dataset will contain 2 columns.  
1. The location column with one of the following column names:  
Country (2-char), State (2-char), Fips (5-digits for state and county), Zip (5 char, 6 in China), or Voxel (2 char)
2. The "Target" column containing 1 or 0

### About setting the model in parameters.yaml

Setting the models parameter to "all" would be the equivalent to "lr,rfc,rbf,svm,mlp,xgboost"  

### About path name shortcuts in parameters.yaml

Default features and targets datasets reside in the "input/[data]/features" and "input/[data]/targets" folders for each data source.

The simplest form of the parameters.yaml would be:

	features: industries
	targets: bees

That's the equivalent to:

	features: industries
	 	path: https://github.com/ModelEarth/realitystream/raw/main/input/industries/features/industries-features.csv
	targets: bees
		path: https://github.com/ModelEarth/bee-data/raw/main/targets/bees-targets-top-20-percent.csv
	models: rbf


The features.path and targets.path will have several shorthand versions and a full version from GitHub:

**short** - bees  
**medium** - /bee-data/targets  
**long** - /bee-data/targets/bees-targets-top-20-percent.csv  
**full** - https://github.com/ModelEarth/bee-data/raw/main/targets/bees-targets-top-20-percent.csv


**Path processing rules:**
If there's no slash / in a path parameter, start from the root of the RealityStream repo.
If the file extension is omitted from a path, append .csv.
For a target value of "bees" build the path "/bee-data/targets/bees-targets-top-20-percent.csv"
Replace a space with -targets- in the path.
So for a target value of "bees increase2024" build the path "/bee-data/targets/bees-targets-increase2024.csv"


