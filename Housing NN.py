import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from zlib import crc32

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedShuffleSplit

################# EXTRA FUNCTIONS #################
# This funciton is used to save images
IMAGES_PATH = Path("images")
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()  # This should be called on plt
    plt.savefig(path, format=fig_extension, dpi=resolution)


# The next 5 lines define the default font sizes
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

################# IMPORT DATA #################
housing = pd.read_csv('housing.csv')

################# BASIC DATA HANDLING #################
### Data Cleaning
# We start by dropping missing values
null_rows_idx = housing.isnull().any(axis=1)
housing.loc[null_rows_idx].head()

housing = housing.dropna(subset=["total_bedrooms"])

housing.loc[null_rows_idx].head()


### Handling numerical and categorical attributes
housing_num = housing.select_dtypes(include=[np.number])
housing_cat = housing[['ocean_proximity']]


### One-hot encoding Categorical data
housing_cat.head(8)

from sklearn.preprocessing import OneHotEncoder

# Initialize the OneHotEncoder with sparse output set to False and handling unknown categories
cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

# Fit and transform the training data
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

# Convert the one-hot encoded array into a DataFrame for easier inspection
housing_cat_df = pd.DataFrame(housing_cat_1hot, columns=cat_encoder.get_feature_names_out(), index=housing.index)

# Display the first few rows to check
housing_cat_df.head()


### Merge back together in a dataframe the numerical and categorical attributes
housing_full = pd.concat([housing_num, housing_cat_df], axis=1)


################# SPLITTING AND SHUFFLING #################

### Preparing income category for stratification
housing_full["income_cat"] = pd.cut(
    housing_full["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5])

### Performing stratified shuffle split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Use .iloc to handle positional indexing
for train_index, test_index in split.split(housing_full, housing_full["income_cat"]):
    strat_train_set = housing_full.iloc[train_index]
    strat_test_set = housing_full.iloc[test_index]

# Drop the "income_cat" column from both sets after splitting
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

### Separate features and the log-transformed target
# For the training set
housing = strat_train_set.drop(["median_house_value"], axis=1)  # Features
housing_labels = strat_train_set["median_house_value"].copy()  # Target

# For the test set
housing_test = strat_test_set.drop(["median_house_value"], axis=1)  # Features
housing_test_labels = strat_test_set["median_house_value"].copy()  # Target


### Pipeline for feature scaling
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics.pairwise import rbf_kernel

# Function for normal transformation
normal_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)

'''
################# TRAINING THE NEURAL NETWORK #################
### Define the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

def create_model(input_shape, learning_rate=0.001):
    # Create model with dropout for regularization
    model = Sequential([
        layers.Dense(64, activation="relu", input_shape=input_shape),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])
    
    # Use a simple optimizer with fixed learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Compile model
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    
    return model


### Set up transformation pipeline

preprocessing = ColumnTransformer([
    ("normal", normal_pipeline, ["total_rooms", "total_bedrooms", "population", "households", "median_income"]),
], remainder="passthrough")

# Prepare the data
housing_prepared = preprocessing.fit_transform(housing)

# Create and train the model
input_shape = (housing_prepared.shape[1],)
model = create_model(input_shape)


# Set early stopping
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)


# Train the model
history = model.fit(
    housing_prepared, 
    housing_labels_log,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Preprocess the test data
housing_test_prepared = preprocessing.transform(housing_test)

# Evaluate the model on the test set
test_loss, test_mae = model.evaluate(housing_test_prepared, housing_test_labels)
print("Test MAE (log-transformed):", test_mae)

# Make predictions on the test set
predictions = model.predict(housing_test_prepared)


################# EVALUATE PERFORMANCE #################
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Mean Absolute Error (MAE)
mae = mean_absolute_error(housing_test_labels, predictions)
print("Mean Absolute Error (MAE):", mae)

# 2. Mean Squared Error (MSE)
mse = mean_squared_error(housing_test_labels, predictions)
print("Mean Squared Error (MSE):", mse)

# 3. Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)

# 4. R-squared (R²)
r2 = r2_score(housing_test_labels, predictions)
print("R-squared (R²):", r2)

################# PLOT RESULTS #################

# Plot training & validation loss
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title("Training and Validation Loss Over Epochs (Log-transformed target)")
plt.show()




# Plot predicted vs actual values
plt.scatter(housing_test_labels, predictions, alpha=0.3)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs Actual Values")
plt.plot([min(housing_test_labels), max(housing_test_labels)],
         [min(housing_test_labels), max(housing_test_labels)], color='red')  # Perfect prediction line
plt.show()
'''


### Model using Keras tuner
import keras_tuner as kt
from keras_tuner import Hyperband
from keras.optimizers import Adam

def build_model(hp):
    """
    Model builder function for keras tuner.
    This function defines the hyperparameter search space.
    """
    model = keras.Sequential()
    
    # Tune number of layers and units per layer
    for i in range(hp.Int('num_layers', min_value=1, max_value=5)):
        units = hp.Int(f'units_{i}', min_value=16, max_value=256, step=32)
        
        # First layer needs input shape
        if i == 0:
            model.add(layers.Dense(units=units, 
                                 activation='relu',
                                 input_shape=(housing_prepared.shape[1],)))
        else:
            model.add(layers.Dense(units=units, activation='relu'))
        
        # Tune dropout rate
        dropout_rate = hp.Float(f'dropout_{i}', min_value=0.0, max_value=0.5, step=0.1)
        model.add(layers.Dropout(dropout_rate))
    
    # Output layer
    model.add(layers.Dense(1))
    
    # Tune learning rate
    learning_rate = hp.Float('learning_rate', 
                           min_value=1e-4, 
                           max_value=1e-2, 
                           sampling='log')
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model

### Set up transformation pipeline
preprocessing = ColumnTransformer([
    ("normal", normal_pipeline, ["total_rooms", "total_bedrooms", "population", "households", "median_income"])
], remainder="passthrough")

# Prepare the data
housing_prepared = preprocessing.fit_transform(housing)

# Preprocess the test data
housing_test_prepared = preprocessing.transform(housing_test)

# Initialize the tuner
tuner = Hyperband(
    build_model,
    objective='val_loss',
    max_epochs=100,
    factor=3,
    directory='keras_tuner',
    project_name='housing_price_prediction'
)

# Define early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Print search space summary
tuner.search_space_summary()

# Perform the search
tuner.search(
    housing_prepared,
    housing_labels,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Rebuild and train the best model
best_model = build_model(best_hps)

# Train the best model
history = best_model.fit(
    housing_prepared,
    housing_labels,
    validation_split=0.2,
    epochs=100,
    callbacks=[early_stopping],
    verbose=1
)

# Print the best hyperparameters
print("\nBest Hyperparameters:")
for param, value in best_hps.values.items():
    print(f"{param}: {value}")

# Evaluate the best model
test_loss, test_mae = best_model.evaluate(housing_test_prepared, housing_test_labels)
print(f"\nTest MAE: {test_mae}")

# Make predictions with the best model
predictions = best_model.predict(housing_test_prepared)

# Calculate metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(housing_test_labels, predictions)
mse = mean_squared_error(housing_test_labels, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(housing_test_labels, predictions)

print("\nFinal Model Performance:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.4f}")

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plot predicted vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(housing_test_labels, predictions, alpha=0.5)
plt.plot([housing_test_labels.min(), housing_test_labels.max()],
         [housing_test_labels.min(), housing_test_labels.max()],
         'r--', lw=2)
plt.xlabel('Actual House Prices')
plt.ylabel('Predicted House Prices')
plt.title('Predicted vs Actual House Prices')
plt.tight_layout()
plt.show()