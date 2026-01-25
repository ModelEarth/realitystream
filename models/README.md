[RealityStream](../)
# RealityStream Forecasting Models

Our Machine Learning (ML) models join feature and target (Y=1) data based on location IDs.  
Scope includes: Countries, States, County FIPS, Zip Codes, and Brain Voxels (for eye blinks).

- [Run Models CoLab](../input/industries)
- [Random Forests for Healthy Bees](../../bee-data/)
- [Random Bits Forest for Eye Blinks](random-bits-forest)

The Run Modles colab merges feature and target datasets in Pandas.  
Training files can be reviewed by setting save_training=True.

Select one or more models, then paste the resulting yaml in the Run Models CoLab.

## <input type="checkbox" id="model-lr" name="model" value="lr"> Logistic Regression (lr)
- **Type**: Linear model for binary classification (extendable to multiclass).
- **Key Feature**: Predicts probabilities using the logistic (sigmoid) function.
- **Best for**: Clean, balanced datasets with approximately linear relationships between features and target.
- **Common Use**: Medical diagnostics, marketing (e.g., churn prediction), financial risk.
- **Limitations**: Struggles with non-linear relationships and complex patterns.

## <input type="checkbox" id="model-rfc" name="model" value="rfc"> Random Forest Classifier (rfc)
- **Type**: Ensemble of decision trees using bootstrapping and random feature selection.
- **Key Feature**: Reduces overfitting and variance through randomness and averaging.
- **Best for**: Diverse data types, handles outliers, missing data, and categorical features.
- **Common Use**: Fraud detection, bioinformatics, credit scoring.
- **Limitations**: Less interpretable, computationally expensive with large datasets.

## <input type="checkbox" id="model-rbf" name="model" value="rbf"> Random Bits Forest (rbf)
- **Type**: Variation of Random Forests using bit-based transformations for feature splitting.
- **Key Feature**: Efficient handling of high-dimensional, binary, or sparse data.
- **Best for**: Cybersecurity, large-scale categorical or binary feature data.
- **Common Use**: High-dimensional binary datasets, one-hot encoded data.
- **Limitations**: May not perform as well on general-purpose or continuous feature datasets.
Overview: [Eye Blink Data](random-bits-forest/) - Load [parameters-blink.yaml](../#parambase=Eye+Blinks)

## <input type="checkbox" id="model-svm" name="model" value="svm"> Support Vector Machines (svm) - Runs Slow

Runs slow (over an hour), best to avoid for now.

- **Type**: Supervised learning algorithm for classification and regression.
- **Key Feature**: Maximizes the margin between classes, supports non-linear relationships via kernel tricks.
- **Best for**: Small to medium-sized, high-dimensional datasets with well-separated classes.
- **Common Use**: Text classification, image recognition, bioinformatics.
- **Limitations**: Struggles with large datasets and noisy data, can be computationally expensive.

## <input type="checkbox" id="model-mlp" name="model" value="mlp"> Neural Network Multi-Layer Perceptron (mlp)
- **Type**: Feedforward artificial neural network with multiple layers.
- **Key Feature**: Learns complex non-linear relationships through backpropagation.
- **Best for**: Large datasets with intricate feature interactions (e.g., images, speech).
- **Common Use**: Deep learning tasks, image processing, speech recognition.
- **Limitations**: Requires large datasets and computational resources, prone to overfitting on small datasets.

## <input type="checkbox" id="model-xgboost" name="model" value="xgboost"> XGBoost (xgboost)
- **Type**: Gradient-boosted decision tree algorithm.
- **Key Feature**: Sequentially corrects previous errors, highly efficient with built-in regularization.
- **Best for**: Large, structured tabular data, handles missing data natively.
- **Common Use**: Financial modeling, fraud detection, machine learning competitions.
- **Limitations**: Complex and harder to interpret, requires tuning for optimal performance.


---
**<button onclick="redirectToMainPage()" class="btn btn-success">Continue</button>**

---

<!--
# Inflow, Outflow, Predicted Results

x-axis Features (naics, voxels, nutrients)  
y-axis Locations merged with target column on county, zip code, or other common attribute.

Features and targets are merged on locations (fips, voxels, foods)

| Inflow | Basket of Goods| Outflow | Predicted Results |
| ----------- | ----------- | ----------- | ----------- |
| [Suppliers](/data-pipeline/research/economy/) | [Commodities](/localsite/info/) | [Products](https://github.com/ModelEarth/profile/tree/main/products/US) | [Impact on Environment](/community/tools/) |
| [Stimulus ML](../blinks/) | Brain Waves | [Brain Voxels Firing](/realitystream/models/random-bits-forest/) | [Eye Blinks](/realitystream/output/blinks/) |
| [Local Industries](/localsite/info/) | Honey Bees | [Population Change](/data-pipeline/research/bees/) | [Healthy Bee Population](/realitystream/output/bees) |
| [Local Industries](/localsite/info/) | [Tree Canopy](/data-commons/docs/conservation/) | Biodiversity Change | Healthy Forest Growth |
| [Ingredients](/data-commons/docs/food/) | [Healthy Meals](/profile/item) | [Nutrients](/balance/) | [Impact on Body](/balance/label_checker.html) |
-->

<!--
We're working with Google Data Commons data to explore trends across time.
[Our BlueSky Projects](https://bsky.app/profile/modelearth.bsky.social) and [Feed Player Displays](https://model.earth/feed/view/) merge industry and environmental data to explore outcomes.

Do Google search algorithms direct people toward training that results in a better world?  Trees grow based on supporting networks of fungi using biological algorithms. Are the locations where people relocate driven by the software they use, and the skills they offer? Does using Facebook, Microsoft, X, Douyin and BlueSky foster similar outcomes? The Google jobs API can be integrated using [Serp](/feed/view/#feed=serp).

[Does expanding access to Starlink actually help increase tree canopy?](https://www.yahoo.com/news/elon-musk-diplomacy-woo-wing-155604090.html) In Brazil, Starlink was slated to provide internet connectivity to 19,000 rural schools, along with environmental monitoring of the Amazon. Let's explore changes to [world forest coverage over time](/data-commons/docs/conservation/).
-->

Paste the resulting yaml parameters list into the second step of the [Run Models CoLab](../input/industries).
