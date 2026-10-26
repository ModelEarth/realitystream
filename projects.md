## Interactive UX

1. Frontend (JavaScript): A web-based interface that allows users to trigger actions and interact with the workflow.
2. Backend (local or Google Cloud Run): A service processes requests from the frontend and execute the python script.

## Projects

Replace TO DO with your name as you work on a project.  
Write Loren when you've submitted a pull request to show your name.

1. DONE: Generate features-importance reports for available models - Melody and Yogesh.
2. DONE: Add a comparison process for accuracy reports into one table for viewing. - Melody
3. DONE: Frontend UX for selecting Features and Targets - Kashmira
4. DONE: Send report files from the local python run to GitHub (`api/run_models.py --upload`, `push_report`; the Run Models page pushes with a browser-saved token). Set `GITHUB_REPORTS_REPO` to push to your own fork of [reports](https://github.com/ModelEarth/reports) when testing. - Vivek
5. PARTIALLY DONE: Add output files to the "[report](https://github.com/ModelEarth/reports/)" folder and add smart formatting in the index.html and javascript report display. We send its content to GitHub as a final step in run_models.py.
6. DONE: Run the Forest Canopy and all-years parameter files through `api/run_models.py` unchanged: join column aliases (`id`, `join.key`, `targets.column`), 5-digit FIPS matching, index-column drop, readable NAICS names in feature importance. - Vivek
7. IN PROGRESS: Automated pull of recent forest canopy data from Google Data Commons (`process/canopy/canopy-process.py`), replacing the manually added target file and the missing timeseries file in [tree-canopy](https://github.com/ModelEarth/tree-canopy). Needs a `DATACOMMONS_API_KEY`. - Vivek
8. TO DO: Decide what the `timeseries` section of a parameters file should drive (extra features per year, a trend target, or a chart in the report). `run_models.py` currently reports whether the file is reachable and does nothing else with it.
9. TO DO: Pull 2-column targets (location id, Target) for UN topics directly from Google Data Commons using `dcid` values in parameters.yaml; `parameters-gdc.yaml` and `parameters_un_gcd.yaml` are the starting points. - Vivek (Nov 5)
10. TO DO: More parameters files that join features and targets on county Fips, including the [all-years](https://github.com/ModelEarth/community-timelines/tree/main/training/all-years) features for states beyond Minnesota (one wide row per county, `Emp-<naics>-<year>` columns).
11. TO DO: Test the default target path for bee data: when a parameters file has no `targets.path` and no `features.target_column`, `load_data` raises; decide whether to default to [bees-targets-top-20-percent.csv](https://github.com/ModelEarth/bee-data/tree/main/targets) as the colab did.
12. TO DO: Compare each model's hold-out metrics with the state-held-out cross-validation added in #66, and show both on the report page so geographic overlap is visible.

Additional history in our [RealityStream project list](https://github.com/modelearth/projects/issues/63). The Run Models colab is no longer maintained; all work happens in `api/run_models.py` and the [Run Models](models/) page.

### Updates made in 2025

Aryaman P - Moved parameter textbox widget near top of former colab and made updates for Forest Canopy data.

Soham D - Repaired the Core GDC Data Function: Replaced the load_gdc_data_if_present function with a robust "V3" version that now correctly expands state geoIds into a full list of child counties and reliably pivots the data into the required wide format. - Soham 

Soham D - Resolved Data Mismatch Bugs: Implemented a fix in the data loading cell to standardize FIPS codes to a 5-digit string format across all data sources, solving the merge failures.

Soham D - Resolved Machine Learning Model Bugs: Fixed a ValueError in SMOTE by setting k_neighbors=4 to handle the small minority class in the training data, and resolved a AttributeError by making the feature importance extraction compatible with both cuML and scikit-learn model objects.

DONE: Include the time it took to run each model in report.md. - TARUN

DONE: Generate features-importance reports for available models. - Bin(Melody)

DONE: Performance metrics—including accuracy, ROC-AUC, G-Mean, best threshold, and classification reports—were aggregated into a modelResults dictionary using abbreviated keys. Top 10 feature importances for applicable models were included, and results were formatted into summary tables. - Yogesh Gajula

DONE: Function to calculate and append Correlation values to Unified Aggregation Results and Visual chart with prefix's for the top 10 Feature importances. - Yogesh Gajula

DONE Aashish: Used Pandas for integrated_df (became df) when save_training = False.  
 
DONE Tarun: Allow save-training to be set in the parameters.yaml values. Default to false. Use dash instead of underscore in yaml.

DONE Ivy: Accuracy report displayModelHeader to display the model name as header and the file paths for features and targets above the report. Parameter values below each path at the top of each accuracy report. So under the Feature path we'd have:  
startyear: 2017, endyear: 2021, naics: [6], state: ME

DONE Lily: Add support for multiple states. After running the third panel, you can edit the custom yaml on the right to set state: CT, ME, MA, NH, RI, VT.  Then add a loop that runs when there are multiple states. We'll add a file called parameters-new-england.yaml in the root of the RealityStream repo with the six states as features.states. Load here and add python to loop through the states.

DONE: Load blinks/parameters-blinks.yaml and use target.column to limit to y column

Done: Avoid sorting incoming parameters.yaml alphabetically. Attempt using  OrderedDict is commented out is several places below. Comment out prior alphabetical technique - we can provide a bool to toggle to it if it provides better security when requests are submitted through webpages. - Soham

DONE: Only import models requested by parameters.yaml. Move "from sklearn" imports to step after parameters are edited in textbox. - Tarun

DONE: Send the params loaded from the default path to the widget diplay. - Prathyusha

DONE: Create an object that holds the 5 sample parameters.yaml paths that are on the RealityStream main page. When choosing one, send the path and the yaml it points at to the textarea below the path select menu. - Prathyusha

DONE: Parameter files displayed in select menu. Instead pull the select options from parameter-paths.csv - Prathyusha

DONE: Deactivate the right-side display of the yaml values and have the editing occur in the widget textbox. - Melody
