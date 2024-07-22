# Demo 2

## Business goal and machine learning solution

### Business Question/Goal
The primary business question being addressed in this project is: "How can we predict customer purchase amounts during Black Friday sales to optimize marketing strategies and increase profits?" The goal is to leverage machine learning to understand and anticipate customer purchasing behavior based on various demographic and behavioral factors. By accurately predicting purchase amounts, businesses can tailor their marketing efforts, allocate resources more efficiently, and ultimately increase sales and profitability during high-traffic shopping events like Black Friday.

### Machine Learning Use Case
The machine learning use case for this project involves developing a predictive model that estimates the purchase amount for each customer based on a set of features. These features include:

### Age
City Category
Gender
Marital Status
Occupation
Product Category 1
Product Category 2
Product Category 3
Stay in Current City Years
By analyzing these features, the model aims to predict the purchase amount each customer is likely to spend during the Black Friday sales.

### How the Machine Learning Solution Addresses the Business Goal
The machine learning solution is designed to provide actionable insights into customer purchasing behavior. Here’s how it addresses the business goal:

* Targeted Marketing Campaigns: By predicting the purchase amounts, businesses can identify high-value customers and target them with personalized marketing campaigns. This increases the likelihood of higher sales conversions.

* Inventory Management: Understanding predicted purchase patterns helps in managing inventory more effectively. Products that are expected to be in high demand can be stocked in advance, reducing the risk of stockouts or overstocking.

* Resource Allocation: Marketing budgets can be allocated more efficiently by focusing on customer segments with higher predicted purchase amounts. This ensures a better return on investment (ROI) for marketing expenditures.

* Customer Insights: The model provides deeper insights into customer behavior and preferences. This can inform future product development and promotional strategies, aligning them more closely with customer needs and preferences.

* Profit Optimization: By optimizing marketing strategies and inventory management, businesses can increase their overall profitability during Black Friday and other peak shopping periods.

### Top-Line Description
This demo presents a machine learning solution aimed at predicting customer purchase amounts during Black Friday sales. The solution leverages demographic and behavioral data to build a predictive model that helps businesses enhance their marketing strategies, optimize resource allocation, and ultimately increase profits. By addressing the business question of how to anticipate customer spending, this project demonstrates the value of machine learning in driving business growth and efficiency during high-traffic shopping events.

## Whitepaper Section: Data Exploration

### Data Exploration Process


Memory updated
Whitepaper Section: Data Exploration
Data Exploration Process
The data exploration process was a critical step in understanding the characteristics of the dataset and making informed decisions about the data preprocessing and model architecture. The exploration was performed using the tensorflow_data_validation (TFDV) library in conjunction with the StatisticGen, SchemaGen, and ExampleValidator components of the TensorFlow Extended (TFX) pipeline. These tools provided comprehensive insights into the dataset's statistics, schema, and potential anomalies

### Tools and Techniques Used

**TensorFlow Data Validation (TFDV):** 

* StatisticGen: This component generated descriptive statistics for the dataset, providing insights into the distribution of values for each feature.
* SchemaGen: This component inferred the schema of the dataset, which includes information about feature types, domains, and expected values.
* ExampleValidator: This component validated the dataset against the inferred schema and identified any anomalies or inconsistencies.

### Type of Data Exploration Performed

1. Descriptive Statistics:

* Examined the mean, median, standard deviation, minimum, and maximum values for numerical features.
* Analyzed the distribution and frequency of categorical features.

2. Schema Inference:

* Identified the data types (e.g., integer, float, string) for each feature.
* Defined domains for categorical features and ranges for numerical features.

3. Anomaly Detection:

* Detected missing values, outliers, and inconsistencies in the dataset.
* Validated that the data adhered to the expected schema.

### Impact on Data/Model Algorithm/Architecture Decisions

The insights gained from the data exploration phase significantly influenced the following aspects of the project:

**Data Preprocessing:**

Implemented normalization for the Purchase feature to handle the wide range of values.
Applied one-hot encoding to categorical features to facilitate their use in the machine learning model.

**Model Architecture:**

Chose algorithms and model architectures that can effectively handle the normalized and encoded features.
Ensured that the model can accommodate the processed data format, leading to better performance and accuracy.

### Conclusion
The data exploration process provided a foundational understanding of the dataset and informed critical decisions regarding data preprocessing and model architecture. By leveraging TFDV and the TFX pipeline components, we ensured that the dataset was thoroughly analyzed and preprocessed, leading to a robust and effective machine learning solution.

## Feature Engineering

Feature engineering is a crucial step in preparing the dataset for machine learning. It involves transforming raw data into meaningful features that improve the performance of the machine learning model. In this project, we performed several feature engineering tasks to convert the raw features into a suitable format for the model.

### Feature Engineering Performed

1. Handling Missing Values:
* We replaced missing values in the dataset with default values to ensure that the model training process is not affected by missing data.

2. One-Hot Encoding:
* We applied one-hot encoding to categorical features to convert them into a numerical format. This process involves creating binary vectors that represent the presence of each category.
* Categorical features were divided into two groups:
  - Categorical Numerical Feature
  - Categorical String Features

3. Normalization:
* We normalized the Purchase feature to ensure that its wide range of values does not bias the model. Normalization helps in scaling the data to have a mean of 0 and a standard deviation of 1.

4. Feature Selection:
* We performed feature selection to focus on the most relevant features for the business use case. We dismissed user_id and product_id as they do not add business value to the prediction of purchase amounts and may not be present at prediction time. This decision helps in simplifying the model and ensuring it generalizes well to unseen data.

### Conclusion

The feature engineering process involved handling missing values, one-hot encoding categorical features, normalizing the purchase label, and performing feature selection. These steps were essential in preparing the data for the machine learning model, ensuring that the features are in a suitable format for training. By transforming the raw data into meaningful features and selecting the most relevant ones, we enhanced the model's ability to accurately predict customer purchase amounts, contributing to more effective marketing strategies and increased profits.

## Data Preprocessing and the Data Pipeline

The data preprocessing pipeline is designed to transform raw data into a format suitable for model training and serving. This pipeline ensures that the data is cleaned, transformed, and standardized, making it ready for the machine learning model. The preprocessing steps are encapsulated in a callable API to enable seamless integration with the production environment where the model will be served.

### Data Ingestion
The data ingestion step loads the raw data into the pipeline using the CsvExampleGen component. This component reads CSV files and splits the data into training, evaluation, and testing sets. The code snippet for this component is stored in `black_friday_pipeline/components/data_ingestion.py` 

### Data Validation
Data validation is performed using the StatisticsGen, SchemaGen, and ExampleValidator components. These components generate statistics, infer the schema, and validate the dataset against the schema to detect any anomalies or inconsistencies. The code snippet for this component is stored in `black_friday_pipeline/components/data_validation.py`

### Data Transformation
The data transformation step involves applying feature engineering techniques such as one-hot encoding for categorical features and normalization for the purchase label. This is accomplished using the Transform component, which ensures that the same transformations are applied during both training and serving.The code snippet for this component is stored in `black_friday_pipeline/components/data_transformation.py`

### Callable API for Data Preprocessing
The preprocessing steps are encapsulated in a function called preprocessing_fn, which is part of the data_transformation.py module. This function is called by the Transform component to apply the necessary transformations to the data. The Transform component ensures that the same preprocessing logic is applied during both training and serving, maintaining consistency and accuracy.

### Integration with Production Model
The preprocessed data is fed into the machine learning model using the Trainer component. The preprocessing function is accessed by the served production model through the TFX Transform component. This integration ensures that the model receives data in the correct format, both during training and when making predictions in production.

### Conclusion
The data preprocessing pipeline involves multiple steps, including data ingestion, validation, and transformation. These steps are encapsulated in a callable API, enabling seamless integration with the production environment. By ensuring consistent data preprocessing during both training and serving, the pipeline contributes to the accuracy and reliability of the machine learning model.



