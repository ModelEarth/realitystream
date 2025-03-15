Our [Run Models CoLab](input/industries) provides Logistic Regression, Support Vector Machines (SVM), MLP, RandomForest, XGBoost 

Our main input is currently industry features by county for exploring environmental impact targets.  
We are also creating [CoLabs for Exiobase International Trade Flow](https://model.earth/profile/trade).


[Run-Models-bkup.ipynb](https://github.com/ModelEarth/RealityStream/tree/main/models) is a backup of the [Run Models CoLab](https://colab.research.google.com/drive/1zu0WcCiIJ5X3iN1Hd1KSW4dGn0JuodB8?usp=sharing) that we run locally. We append "-bkup" to indicate it is not the primary source.

<h2>Design your Stream</h2>

**Bee YAML Updated** - Changed [bee data](input/bees) target to bees-targets-top-20-percent.csv in parameters yaml. This new "colony density" target uses the top 20% of counties with the highest bee population density (rather than top colony growth between years, as was used in bees-targets.csv).

<!--
Density file: bees-targets-top-20-percent.csv. Shashank worked from bees-population-usda.csv
(We previously used growth over time with the file bees-targets.csv)
-->

Copy and paste one of the following or use the CoLabs default settings:
[parameters-simple.yaml](https://raw.githubusercontent.com/ModelEarth/RealityStream/main/parameters/parameters-simple.yaml) - 2020, just Maine
[parameters.yaml](https://raw.githubusercontent.com/ModelEarth/RealityStream/main/parameters/parameters.yaml) - Predicts bee density by industry  
[parameters-years.yaml](https://raw.githubusercontent.com/ModelEarth/RealityStream/main/parameters/parameters-years.yaml) - For testing with multiple years and states (currently same as parameters.yaml).  Uses bee populatin growth.
[parameters-zip.yaml](https://raw.githubusercontent.com/ModelEarth/RealityStream/main/parameters/parameters-zip.yaml) - Needs zip code target. Uses bee populatin growth.  
[parameters-blinks.yaml](https://raw.githubusercontent.com/ModelEarth/RealityStream/main/parameters/parameters-blinks.yaml) - Uses only features dataset (which contains the target column).

# Local Setup

### Prerequisites
- Git
- Python 3.8+
- pip (Python package manager)
- Homebrew (for macOS users)

### Initial Setup

1. Clone the repository
   ```bash
   git clone https://github.com/cvnad1/RealityStream.git
   ```

2. Navigate to the project directory
   ```bash
   cd RealityStream
   ```

3. Create a feature branch
   ```bash
   git checkout -b feature_branch
   ```
   Replace `feature_branch` with a descriptive name related to your work (e.g., `user-authentication`, `video-streaming`).

4. Create a virtual environment
   ```bash
   python3 -m venv ./venv
   ```

5. Activate the virtual environment
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

6. For macOS users: Install libomp
   ```bash
   brew install libomp
   ```
   **Note:** libomp is required for certain machine learning libraries. Some users might encounter an error related to OpenMP when installing dependencies. This step preemptively solves that issue.

7. Install required dependencies
   ```bash
   pip3 install -r requirements.txt
   ```

## Git Workflow

### Before Starting Development

Always pull the latest changes from the main branch before starting your work:

```bash
git pull origin main
```

### Adding New Packages

If you install new Python packages:

1. Add them to your development environment:
   ```bash
   pip3 install package_name
   ```

2. Update the requirements.txt file:
   ```bash
   pip freeze > requirements.txt
   ```

### Committing Changes

1. Check the status of your changes:
   ```bash
   git status
   ```

2. Add your changes to the staging area:
   ```bash
   git add .
   ```
   Or to add specific files:
   ```bash
   git add file_name
   ```

3. Commit your changes with a descriptive message:
   ```bash
   git commit -m "Add feature X" 
   ```

4. Push your changes to the remote repository:
   ```bash
   git push origin feature_branch
   ```

### Creating a Pull Request

After pushing your changes, go to the GitHub repository and create a pull request to merge your feature branch into the main branch.

## Contributing

1. Always work on a feature branch, never directly on main
2. Keep commits small and focused
3. Write clear commit messages
4. Update documentation as needed
5. Make sure all tests pass before submitting a pull request
