# Demo 2

## Table of Contents
1. [Business Goal and Machine Learning Solution](#business-goal-and-machine-learning-solution)
   - [Business Question/Goal](#business-questiongoal)
   - [Machine Learning Use Case](#machine-learning-use-case)
   - [How the Machine Learning Solution Addresses the Business Goal](#how-the-machine-learning-solution-addresses-the-business-goal)
   - [Top-Line Description](#top-line-description)
2. [Data Exploration](#data-exploration)
   - [Data Exploration Process](#data-exploration-process)
   - [Tools and Techniques Used](#tools-and-techniques-used)
   - [Type of Data Exploration Performed](#type-of-data-exploration-performed)
   - [Impact on Data/Model Algorithm/Architecture Decisions](#impact-on-datamodel-algorithmarchitecture-decisions)
   - [Conclusion](#conclusion)
3. [Feature Engineering](#feature-engineering)
   - [Feature Engineering Performed](#feature-engineering-performed)
   - [Conclusion](#conclusion-1)
4. [Data Preprocessing and the Data Pipeline](#data-preprocessing-and-the-data-pipeline)
   - [Data Ingestion](#data-ingestion)
   - [Data Validation](#data-validation)
   - [Data Transformation](#data-transformation)
   - [Callable API for Data Preprocessing](#callable-api-for-data-preprocessing)
   - [Integration with Production Model](#integration-with-production-model)
   - [Conclusion](#conclusion-2)
5. [Machine Learning Model Design and Selection](#machine-learning-model-design-and-selection)
   - [Machine Learning Model Selection](#machine-learning-model-selection)
   - [Model Design and Training](#model-design-and-training)
6. [Machine Learning Model Evaluation](#machine-learning-model-evaluation)
   - [Evaluation Process](#evaluation-process)
   - [Evaluation Metrics](#evaluation-metrics)
   - [Evaluation Results](#evaluation-results)
7. [Fairness Analysis](#fairness-analysis)
   - [Fairness Evaluation](#fairness-evaluation)
   - [Mitigating Bias](#mitigating-bias)
   - [Conclusion](#conclusion-3)

## Business goal and machine learning solution

### Business Question/Goal
The primary business question being addressed in this project is: "How can we predict customer purchase amounts during Black Friday sales to optimize marketing strategies and increase profits?" The goal is to leverage machine learning to understand and anticipate customer purchasing behavior based on various demographic and behavioral factors. By accurately predicting purchase amounts, businesses can tailor their marketing efforts, allocate resources more efficiently, and ultimately increase sales and profitability during high-traffic shopping events like Black Friday.

### Machine Learning Use Case
The machine learning use case for this project involves developing a predictive model that estimates the purchase amount for each customer based on a set of features. These features include:
* Age
* City Category
* Gender
* Marital Status
* Occupation
* Product Category 1
* Product Category 2
* Product Category 3
* Stay in Current City Years

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

## Data Exploration

### Data Exploration Process

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

## Machine Learning Model Design and Selection

### Machine Learning Model Selection
For this project, we selected a Convolutional Neural Network (CNN) model for predicting the purchase amounts during Black Friday sales. The decision to use a CNN model was based on several criteria aimed at achieving high accuracy and robustness in predictions.

#### Criteria for Model Selection

1. Data Characteristics:

- The dataset contains various categorical and numerical features that can benefit from the hierarchical feature extraction capabilities of CNNs.

2. Predictive Performance:

- CNNs are known for their ability to capture complex patterns and interactions in the data, which is essential for accurately predicting purchase amounts.

3. Scalability:

- The model needs to handle large volumes of data efficiently. CNNs, with their parallel processing capabilities, are well-suited for this requirement.

4. Previous Success:

- CNNs have been successfully applied in similar use cases, providing a strong justification for their selection in this project.

5. Ease of Integration:

- The model should be easily integrable into the existing TFX pipeline, allowing for seamless data preprocessing, training, evaluation, and serving.

### Model Design and Training
The model design and training process involves defining the CNN architecture, compiling the model, and training it using the transformed dataset. Key aspects of the training process include:

1.  Model Architecture:

  - The CNN model consists of multiple dense layers to capture the complex relationships between the input features.
  - Input layers are created based on the transformed feature specifications.
  - Dense layers with ReLU activation functions are used to introduce non-linearity and learn complex patterns.
  - The output layer is a single neuron that predicts the purchase amount.

2. Optimizer and Learning Rate:

  - The Adam optimizer is used for training the model due to its efficiency and adaptability in handling sparse gradients.
  - An exponential decay schedule is applied to the learning rate, starting at 0.1 and decaying by a factor of 0.9 every 1000 steps. This helps in stabilizing the training process and improving convergence.

3. Early Stopping:

  - Early stopping is implemented to monitor the validation loss and stop training if the model's performance does not improve for 5 consecutive epochs. This prevents overfitting and saves computational resources.

4. Training Steps:

  - The model is trained for a maximum of 50,000 steps with an evaluation step every 10,000 steps. This ensures that the model is adequately trained and evaluated periodically.

5. Callbacks:

  - TensorBoard callbacks are used to monitor the training process and log metrics for visualization.

The code snippet for model design and training can be found on `black_friday_pipeline/components/model_trainer.py`


### Model Evaluation and Pushing
The evaluation of the trained model is performed using the Evaluator component, which measures the model's performance on the test dataset. The evaluation criteria include metrics such as Root Mean Squared Error (RMSE). The code snippet for model evaluation can be found on `black_friday_pipeline/components/model_evaluator_and_pusher.py`



## Machine learning model evaluation

After training and optimizing the machine learning model, it is crucial to evaluate its performance on an independent test dataset. This ensures that the model generalizes well to new, unseen data, which reflects the distribution it is expected to encounter in a production environment.

### Evaluation Process
The evaluation process involves several steps:

1. Evaluation Configuration:

  - An evaluation configuration is set up to specify the evaluation metrics and slicing specifications. For this project, the primary metric used is Root Mean Squared Error (RMSE), which is appropriate for regression tasks.

2. Model Resolver:

  - A model resolver is used to ensure that the latest blessed model is selected as the baseline for comparison during evaluation. This allows for a continuous improvement cycle by comparing new models against the best-performing previously deployed models.

3. Evaluator Component:

  - The Evaluator component of TFX is used to assess the model's performance on the independent test dataset. This component computes the specified metrics and generates detailed evaluation reports.

4. Independent Test Dataset:

  - The model is evaluated on an independent test dataset that reflects the distribution of data expected in a production environment. This dataset is kept separate from the training and validation datasets to provide an unbiased assessment of the model's performance.

### Evaluation Metrics
The primary evaluation metric for this project is Root Mean Squared Error (RMSE). RMSE measures the average magnitude of the errors between the predicted and actual purchase amounts, providing a clear indication of the model's predictive accuracy.

### Evaluation Results
The evaluation results are derived from the Evaluator component and provide insights into how well the model performs on the independent test dataset. The key metric, RMSE, is used to measure the prediction accuracy. If the model has a better result in the key metric that the one defined on the threshold, the Evaluator "blesses" the model and the Pusher component registers and deployes it in an Vertex AI endpoint. The code for the Evaluator and Pusher are stored in `black_friday_pipeline/components/model_evaluator_and_pusher.py`.

## Fairness Analysis

The machine learning model developed for predicting purchase amounts on the Black Friday dataset, intended for targeted marketing, inherently includes purchaser demographics such as "Age," "Gender," and "City_Category." While these features can enhance the accuracy of predictions, they also pose potential fairness and bias issues. Specifically, the model may disproportionately favor or disadvantage certain demographic groups, leading to biased marketing strategies.

### Fairness Evaluation

1. What Was Evaluated?

  - The model's performance was evaluated across different demographic slices, particularly focusing on "Age," "Gender," and "City_Category." Metrics such as Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) were calculated for each slice to assess the model's accuracy and potential bias.

2. Why Evaluate These Slices?

  - Evaluating these demographic slices helps identify if the model performs unevenly across different groups, which is crucial for ensuring fair and unbiased predictions. Uneven performance could indicate that the model is biased towards or against specific demographics, leading to unfair treatment in targeted marketing strategies.

3. Findings:

  - The fairness evaluation revealed that the model's performance was generally consistent across most demographic groups. However, it was found that the model exhibited higher error rates (both RMSE and MAE) for the age groups "0-17" and "55+." These groups were underrepresented in the training data, leading to less accurate predictions for these demographics.

### Mitigating Bias

To address the identified bias against the "0-17" and "55+" age groups, the following steps were implemented:

1. What Was Done?
  - ***Weight Sampling***: To compensate for the underrepresentation of the "0-17" and "55+" age groups in the training data, weight sampling was applied. This technique assigns higher weights to these underrepresented groups during the training process, ensuring that the model pays more attention to these samples and learns more effectively from them.

2. Why Use Weight Sampling?

  - Weight sampling is a practical method to address data imbalance by giving more importance to underrepresented groups. This helps in reducing the disparity in model performance across different demographic slices, promoting fairness and equity in the model's predictions.

3. How Was It Implemented?

  - The implementation involved adjusting the sample weights during the training process, as illustrated in the following script:
```
def add_sample_weights(features, label):
            # Extract the 'Age_xf' one-hot encoded feature
            age_one_hot = features['Age_xf']

            # Determine the index of the active age category in the one-hot vector
            age_index = tf.argmax(age_one_hot, axis=1, output_type=tf.int32)

            # Apply weights based on conditions
            sample_weight = tf.where(
                tf.equal(age_index, AGE_GROUP_INDICES['0-17']),
                tf.constant(AGE_GROUP_WEIGHTS[0], dtype=tf.float32),
                tf.where(
                    tf.equal(age_index, AGE_GROUP_INDICES['55+']),
                    tf.constant(AGE_GROUP_WEIGHTS[6], dtype=tf.float32),
                    tf.constant(1.0, dtype=tf.float32)  # Default weight for other groups
                )
            )
            return features, label, sample_weight
```
This function is part of the trainer component defined in `black_friday_pipeline/components/model_trainer.py`

### Conclusion
The fairness analysis revealed that the model initially exhibited bias against the age groups "0-17" and "55+" due to their underrepresentation in the dataset. By implementing weight sampling, the model's accuracy for these groups improved, leading to fairer and more equitable predictions. This approach ensures that targeted marketing strategies do not disproportionately disadvantage any demographic group, aligning the model's outcomes with ethical standards and promoting customer trust.

This thorough fairness analysis and mitigation strategy demonstrate a commitment to responsible AI practices, ensuring that the model's deployment in a production environment aligns with ethical and regulatory standards.
