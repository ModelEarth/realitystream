Our [Run Models CoLab](input/industries) provides Logistic Regression, Support Vector Machines (SVM), MLP, XGBoost and two Random Forests.

Paths for **features** and **targets** are joined based on location IDs (state or county FIPS, etc.) or brain regions for [eye blink predictions](models/random-bits-forest/).

For Features, the UI picks `features.dcid` if it exists otherwise `features.data` otherwise the filename from `features.path`.  
For Targets, it picks `targets.dcid` if it exists otherwise `targets.data` otherwise the filename from `targets.path`.

Our main input is currently industry features by county ID (FIPS) for exploring environmental impact targets like [bee data](../bee-data) and [tree canopy](https://model.earth/tree-canopy/).  
We are also preping data for [International Trade Flow](https://model.earth/profile/trade).


[Run-Models-bkup.ipynb](https://github.com/ModelEarth/realitystream/tree/main/models) is a backup of the [Run Models CoLab](https://colab.research.google.com/drive/1zu0WcCiIJ5X3iN1Hd1KSW4dGn0JuodB8?usp=sharing) that we run locally. We append "-bkup" to indicate it is not the primary source.

Learn about our [cuML GPU speed enhancements - and SMOTE balancing of our classes](cuML)
We're using [SHAP to explain our model predictions](shap)

<h2>Design your Stream</h2>

**Bee YAML Updated** - Changed [bee data](/bee-data) target to bees-targets-top-20-percent.csv in parameters yaml. This new "colony density" target uses the top 20% of counties with the highest bee population density (rather than top colony growth between years, as was used in bees-targets.csv).

<!--
Density file: bees-targets-top-20-percent.csv. Shashank worked from bees-population-usda.csv
(We previously used growth over time with the file bees-targets.csv)
-->

In the CoLab, a select menu allows you to choose default parameter yaml paths set in [parameter-paths.csv](https://github.com/ModelEarth/realitystream/blob/main/parameters/parameter-paths.csv).
[parameters-simple.yaml](https://raw.githubusercontent.com/ModelEarth/realitystream/main/parameters/parameters-simple.yaml) - Bee density for 2020 for Maine and New York by top industries (2-digits)
[parameters.yaml](https://raw.githubusercontent.com/ModelEarth/realitystream/main/parameters/parameters.yaml) - Predicts bee density by industry (6-digit NAICS) for New England states 2017 to 2021
[parameters-gdc.yaml](https://raw.githubusercontent.com/ModelEarth/realitystream/main/parameters/parameters-gdc.yaml) - Google Data Commons - Predicting states using demographics
[parameters-zip.yaml](https://raw.githubusercontent.com/ModelEarth/realitystream/main/parameters/parameters-zip.yaml) - Needs zip code target. Uses bee population density.  
[parameters-blinks.yaml](https://raw.githubusercontent.com/ModelEarth/realitystream/main/parameters/parameters-blinks.yaml) - Uses only features dataset (which contains the target column).

<!--

[parameters-years.yaml](https://raw.githubusercontent.com/ModelEarth/realitystream/main/parameters/parameters-years.yaml) - For testing with multiple years and states (currently same as parameters.yaml).

TO DO: Web page displaying US counties at risk of increased poverty - Use Google Data Commons API for FIPS county poverty target data and international target data. Pull with an [Observable Data Loader](../../../timelines/observable/)
-->

TO DO: Top ten counties in each state likely to have [declining tree canopy](/data-pipeline/research/canopy/)

TO DO: Within Run Models, add python pull from Google Data Commons API for population, education levels, income/poverty levels to use as both features and targets. - IN PROGRESS

TO DO: Use [Tensorflow.org](https://www.tensorflow.org/js/demos) for [Neural Network predictions](https://www.tensorflow.org/s/results/?q=neural%20networks) with our training data.

<!--
### Javascript Display in Tabulator

In javascript, we'll populate "Density" for each county and append it as a column in Tabulator. [Tabulator work in progress](/data-pipeline/timelines/tabulator/).

Density = Population / Km2

Density can also be thought of as PopPerKm2 (divided by 1000)
100,000 people living in an 80 Km2 county = 1250 people per Km2 = Density of 1.25
When displaying, we will multiply Density and Population by 1000.

Append 0 or 1 to the "y" column. Prior y column in community forecasting: y=1 when the current year’s poverty had no decline from the prior year AND the next year’s poverty increased by 2% or more.


Applied in
prep/all/zcta_2016.SQL.txt

-- Change from prior year is steady (0%) or increasing, change to next year is increasing by 2% or more.

CASE
      WHEN (prior1.poverty - p.poverty) >= 0 AND (p.poverty - next.poverty) >= 2 THEN 1
      ELSE 0
END

AS y -- the povertyBinary for >= 2% in coming year, and no decline for current year.
-->
