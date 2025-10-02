# Network Intrusion Detection System for Cisco Security
kaggle link::https://www.kaggle.com/code/bikrambarman9932/intrusion-detection

With the rapid growth of digital networks and increasing internet usage, network traffic has surged, leading to heightened risks of cyber threats such as unauthorized access, DoS attacks, and data breaches. Intrusion Detection Systems (IDS) are essential for identifying and mitigating these threats to protect network infrastructure. This project develops a Cisco-aligned IDS using the [KDD Cup 1999 dataset](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html), leveraging unsupervised machine learning (Isolation Forest, Autoencoders) for anomaly detection and a supervised XGBoost classifier for benchmark performance. The system includes advanced exploratory data analysis (EDA) to uncover network traffic patterns, designed to integrate with Cisco’s AI-driven security solutions like Cisco Secure Network Analytics and Cisco DNA Center.

The objective is to classify network connections as normal or malicious, detecting both known and novel threats with low false positives. The implementation follows four key steps:
- Load and preprocess the KDD dataset for model compatibility.
- Conduct exploratory data analysis with comprehensive visualizations.
- Train and evaluate unsupervised models (Isolation Forest, Autoencoders) and a supervised XGBoost model.
- Assess performance using accuracy, precision, recall, and false positive metrics.


## Project Steps

### Data Preprocessing
Preprocessing transforms the KDD dataset into a clean, model-ready format:
- **Data Loading**: Imports the `kddcup.data_10_percent.gz` dataset (~494,021 records).
- **Cleaning**: Removes constant features (e.g., `num_outbound_cmds`) and handles outliers.
- **Transformation**: Encodes categorical features (`protocol_type`, `service`, `flag`) using `LabelEncoder` and scales numerical features (e.g., `src_bytes`, `dst_bytes`) with `StandardScaler`.
- **Label Encoding**: Converts `intrusion_type` to binary labels (0 for normal, 1 for malicious) for evaluation.

This ensures the dataset’s diverse features (continuous, discrete, categorical) are suitable for machine learning, aligning with Cisco’s need for robust data pipelines in network security analytics.

### Exploratory Data Analysis (EDA)
EDA uses visualizations to reveal insights into network traffic patterns:
- **Class Distribution Plot**: Bar plot (log scale) showing the imbalance between normal and malicious connections.
- **Violin Plots**: Distributions of `src_bytes` and `dst_bytes` across intrusion types.
- **Correlation Heatmap**: Highlights strong correlations (|corr| ≥ 0.7) among numerical features.
- **Feature Histograms**: Distributions of numerical features (e.g., `duration`, `count`).
- **Pair Plots**: Bivariate relationships for feature groups (e.g., `duration`, `src_bytes`, `root_shell`, `dst_host_serror_rate`).
- **t-SNE Visualization**: 2D projection of high-dimensional data to identify clusters of normal vs. malicious traffic.

These visualizations uncover patterns like high `src_bytes` variance in attacks, informing model design for Cisco’s data-driven security solutions.

### Model Development
#### Isolation Forest
Isolation Forest, an unsupervised algorithm, isolates anomalies by random partitioning, offering efficiency and low false positives, ideal for detecting novel threats in Cisco’s Secure Network Analytics.

#### Autoencoders
Autoencoders, built with TensorFlow, learn to reconstruct normal traffic patterns. High reconstruction errors flag anomalies, making them suitable for identifying unknown attacks in Cisco’s adaptive security framework.

#### XGBoost
XGBoost, a supervised boosting algorithm, constructs sequential decision trees for high-accuracy classification. It serves as a benchmark to compare against unsupervised models, aligning with Cisco’s need for reliable performance.

### Model Evaluation
The IDS classifies connections as normal or malicious:
- **Unsupervised Models**: Isolation Forest and Autoencoders detect anomalies without labeled data, critical for novel threat detection.
- **Supervised Model**: XGBoost leverages labeled data for high accuracy.
- **Metrics**: Evaluated using accuracy, precision, recall, F1-score, and false positives via `classification_report` and `confusion_matrix`, with a focus on minimizing false positives for Cisco’s security operations.

## Results
The IDS achieves robust performance:
- **Isolation Forest**: Precision ~0.85–0.95 for malicious class, low false positives.
- **Autoencoder**: F1-score ~0.75–0.85, effective for novel anomalies.
- **XGBoost**: Accuracy ~99%, false positives ~30–50, comparable to ensemble methods.

The EDA highlights key patterns, such as high `src_bytes` variance in DoS attacks, enhancing model interpretability. The system’s low false positives and ability to detect novel threats align with Cisco Secure Network Analytics’ requirements for real-time, reliable threat detection.



## How to Run
1. **Set Up Virtual Environment**:
   ```bash
   pip install virtualenv
   virtualenv cisco-ids-env
   source cisco-ids-env/bin/activate
   ```
2. **Clone the Repository**:
   ```bash
   git clone <your-repository-url>
   cd <repository-name>
   ```
3. **Install Dependencies**:
   ```bash
   pip install pandas numpy scikit-learn tensorflow xgboost matplotlib seaborn
   ```
4. **Download the Dataset**:
   - Download `kddcup.data_10_percent.gz` from [http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz](http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz).
   - Place it in the project directory.
5. **Run the Notebook**:
   ```bash
   jupyter notebook
   ```
   - Open `cisco_ids_anomaly_detection_with_eda.ipynb`.
   - Update the `file_path` variable to match the dataset location.
   - Run all cells to perform preprocessing, EDA, model training, and evaluation.


## Future Enhancements
- Integrate with Cisco APIs for real-time NetFlow data processing.
- Deploy models in Cisco’s cloud-based ML pipelines for scalability.
- Incorporate Cisco-specific telemetry features (e.g., NetFlow attributes).
- Explore hybrid unsupervised-supervised ensembles for improved accuracy.

## Contributing
For issues or contributions, please open a pull request or contact the repository owner. This project is designed to showcase skills for a Machine Learning Engineer role at Cisco, emphasizing network security and AI-driven analytics.
