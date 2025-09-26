# 🏠 Minerva Housing Price Prediction

A machine learning project that predicts California housing prices using neural networks with automated hyperparameter tuning. This project was created as part of a series of articles for the **MINERVA Student Association** and demonstrates advanced ML techniques including Keras Tuner, stratified sampling, and comprehensive data preprocessing.

> **Note**: This project is based on the housing price prediction example from the book "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" by Aurélien Géron. It was developed to compare neural network approaches with Bayesian Linear Regression and Lasso-Ridge regressions as part of MINERVA's educational content series.

## 📊 Dataset

The project uses the **California Housing Dataset** with the following features:
- **Size**: 20,640 records
- **Features**: 9 input features + 1 target variable
- **Target**: `median_house_value` (housing prices in USD)

### Features
- `longitude`, `latitude` - Geographic coordinates
- `housing_median_age` - Age of the house
- `total_rooms`, `total_bedrooms` - Room counts
- `population`, `households` - Demographics
- `median_income` - Income level
- `ocean_proximity` - Categorical feature (distance from ocean)

## 🚀 Features

- **🤖 Automated Hyperparameter Tuning**: Uses Keras Tuner with Hyperband algorithm
- **📈 Dynamic Neural Network Architecture**: 1-5 layers with 16-256 units each
- **🔧 Robust Data Preprocessing**: Handles missing values, categorical encoding, and feature scaling
- **📊 Stratified Sampling**: Ensures representative train/test splits based on income categories
- **📈 Comprehensive Visualization**: Generates multiple plots for data exploration and model evaluation
- **💾 Model Persistence**: Saves trained models and tuning results
- **🔬 Comparative Analysis**: Part of a series comparing Neural Networks vs Bayesian LR vs Lasso-Ridge regressions

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- TensorFlow 2.x
- scikit-learn
- pandas
- numpy
- matplotlib
- keras-tuner

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/Minerva-Housing-NN.git
cd Minerva-Housing-NN

# Install dependencies
pip install tensorflow scikit-learn pandas numpy matplotlib keras-tuner

# Run the project
python "Housing NN.py"
```

## 📁 Project Structure

```
Minerva-Housing-NN/
├── Housing NN.py              # Main script
├── housing.csv                # California housing dataset
├── images/                    # Generated visualizations
│   ├── housing_prices_scatterplot.png
│   ├── income_vs_house_value_scatterplot.png
│   ├── scatter_matrix_plot.png
│   ├── attribute_histogram_plots.png
│   └── ...
├── keras_tuner/              # Hyperparameter tuning results
│   └── housing_price_prediction/
│       ├── *.json            # Configuration files
│       └── *.h5              # Model files
└── README.md                 # This file
```

## 🔄 Workflow

### 1. Data Preprocessing
- **Missing Values**: Drops rows with missing `total_bedrooms`
- **Categorical Encoding**: One-hot encodes `ocean_proximity` using `OneHotEncoder`
- **Feature Scaling**: Standardizes numerical features using `StandardScaler`
- **Stratified Splitting**: Uses income categories to ensure representative train/test splits

### 2. Hyperparameter Tuning
- **Algorithm**: Hyperband for efficient hyperparameter search
- **Search Space**:
  - Number of layers: 1-5
  - Units per layer: 16-256 (step 32)
  - Dropout rates: 0.0-0.5 (step 0.1)
  - Learning rate: 1e-4 to 1e-2 (log scale)

### 3. Model Training
- **Architecture**: Dynamic neural network based on hyperparameter search
- **Regularization**: Dropout layers to prevent overfitting
- **Early Stopping**: Prevents overfitting with patience=10
- **Validation**: 20% split for validation during training

### 4. Evaluation
- **Metrics**: MAE, MSE, RMSE, R²
- **Visualization**: Training curves and prediction scatter plots

## 📈 Results

The model generates several visualizations:
- **Training History**: Loss curves over epochs
- **Prediction Accuracy**: Scatter plot of predicted vs actual values
- **Data Exploration**: Histograms and correlation matrices
- **Geographic Analysis**: Housing prices by location

## 🎯 Usage

### Basic Usage
```python
# Simply run the main script
python "Housing NN.py"
```

### Custom Configuration
You can modify the hyperparameter search space in the `build_model()` function:

```python
def build_model(hp):
    # Adjust search space as needed
    for i in range(hp.Int('num_layers', min_value=1, max_value=5)):
        units = hp.Int(f'units_{i}', min_value=16, max_value=256, step=32)
        # ... rest of the model building
```

## 📊 Model Performance

The model is evaluated using multiple metrics:
- **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual values
- **Root Mean Squared Error (RMSE)**: Square root of average squared differences
- **R-squared (R²)**: Proportion of variance explained by the model

## 🔧 Dependencies

```
tensorflow>=2.0.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
keras-tuner>=1.0.0
```

## 📝 Notes

- The project includes both a simple neural network approach (commented out) and an advanced hyperparameter-tuned approach
- All visualizations are automatically saved to the `images/` directory
- Hyperparameter tuning results are saved in the `keras_tuner/` directory
- The model uses early stopping to prevent overfitting
- This implementation is part of a comparative study series for MINERVA Student Association
- Based on the housing price prediction example from "Hands-On Machine Learning" by Aurélien Géron

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Aurélien Géron** - "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" (2nd Edition)
- **MINERVA Student Association** - For providing the educational context and series framework
- **California Housing Dataset** - Original dataset source
- **TensorFlow/Keras team** - Deep learning framework
- **scikit-learn contributors** - Machine learning library
- **Keras Tuner developers** - Hyperparameter optimization tool

---

**Happy Predicting! 🏠📈**
