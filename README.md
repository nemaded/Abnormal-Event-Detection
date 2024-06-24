Abnormal Event Detection in Videos Using Modified Spatio-Temporal Autoencoder
**Overview**
This project implements a novel deep learning framework for real-time anomaly detection in crowd event videos. Using a combination of convolutional autoencoders and ConvLSTM2D layers, the system captures both spatial and temporal features, allowing for accurate detection of abnormal events in videos.

**Features**
Real-time anomaly detection in video streams
Utilizes convolutional autoencoders and ConvLSTM2D layers
Advanced data preprocessing techniques
Optimized model training with TensorFlow and Keras
Scalable and domain-free deployment
Performance
Achieved 91% validation accuracy
Processed datasets with over 30,000 frames
Reduced computation time by 50% using GPU acceleration
Publication
This research was published in the International Journal of Advanced Research in Computer and Communication Engineering (IJARCCE), with an impact factor of 6.982.

**Installation
**Clone the repository:

**bash**
Copy code
git clone https://github.com/yourusername/abnormal-event-detection.git
cd abnormal-event-detection
Install the required packages:

**bash**
Copy code
pip install -r requirements.txt
Usage
Preprocessing
Extract frames from video files:

**python**
Copy code
python preprocessing.py --video_path <path_to_video> --output_dir <path_to_output_directory>
Training
Train the model with the preprocessed data:

**python**
Copy code
python train.py --data_path <path_to_preprocessed_data> --epochs 50 --batch_size 16
Deployment
Deploy the trained model for real-time detection:

**python**
Copy code
python DeployModel.py --model_path <path_to_trained_model> --video_path <path_to_video> --threshold 0.5
**Project Structure
**DeployModel.py: Script for deploying the trained model for real-time or video-based anomaly detection.
ModelWrapper.py: Contains functions for building and configuring the deep learning model.
preprocessing.py: Handles the preprocessing of video data, including frame extraction and normalization.
train.py: Manages the training process of the deep learning model.
requirements.txt: Lists all the dependencies required to run the project.
**Dataset**
The Avenue Dataset is used, containing 16 training and 21 testing video clips with a total of 30,652 frames. The videos are captured in CUHK campus avenue and include various challenges such as slight camera shake and outliers.

**Results**
Accuracy: 100% validation accuracy on the training dataset.
Precision: Varied precision based on different anomalies.
Recall: High recall rate for detected anomalies.
**Contributions**
Contributions are welcome! Please fork the repository and submit pull requests.
