## Interactive UX

To send parameters to the machine learning Colab notebook, we're building frontends for a backend Flask server.

1. Frontend (JavaScript): A web-based interface that allows users to trigger actions and interact with the workflow.
2. Backend (Flask or similar): A service to process requests from the frontend and execute the Colab notebook's logic.
3. Colab Notebook: A machine learning notebook hosted in Google Colab or converted into an executable Python script for integration.


## Projects

Replace TO DO with your name as you work on a project.  
Write Loren when you've submitted a pull request to show your name.  
Update related .ipynb and app.py file to also add your name.

1. DONE: Generate features-importance reports for available models - Melody and Yogesh.
2. DONE: Add a comparison process for accuracy reports into one table for viewing. - Melody
3. IN PROGRESS: Frontend UX for selecting Features and Targets - Kashmira
4. TO DO: Send files from local python backup to Github (See run_models_cli.py and run_models.py) <!-- Sampreethi -->

Additional TO DO's reside at the top of the [Run Models colab](https://colab.research.google.com/drive/1zu0WcCiIJ5X3iN1Hd1KSW4dGn0JuodB8?usp=sharing).


## Contribute to our Run Models Colab

[Run Models Colab](input/industries/)

PARTIALLY DONE: Add output files to the "[report](https://github.com/ModelEarth/reports/)" folder and add smart formatting in the index.html and javascript report display. We send it's content to GitHub in a step toward the end.

IN PROGRESS: Creating install for Flask application with Google Cloud Run cmds at [github.com/modelearth/cloud](https://github.com/modelearth/cloud)

TO DO: More item are on our [RealityStream project list](https://github.com/modelearth/projects/issues/63)

<!--
TODO: imblearn import for cuML - Check if already done.
-->

### October 2025

Aryaman P - Moved parameter textbox widget near top of colab and custom steps to Forest Canopy.

Soham D - Repaired the Core GDC Data Function: Replaced the load_gdc_data_if_present function with a robust "V3" version that now correctly expands state geoIds into a full list of child counties and reliably pivots the data into the required wide format. - Soham 

Soham D - Resolved Data Mismatch Bugs: Implemented a fix in the data loading cell to standardize FIPS codes to a 5-digit string format across all data sources, solving the merge failures.

Soham D - Resolved Machine Learning Model Bugs: Fixed a ValueError in SMOTE by setting k_neighbors=4 to handle the small minority class in the training data, and resolved a AttributeError by making the feature importance extraction compatible with both cuML and scikit-learn model objects.

### September 2025 and prior

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
