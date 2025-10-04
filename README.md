# Casting-Product-Defects-detection
AI-powered defect detection system for manufacturing quality control. Uses deep learning (ResNet18) to classify metal casting products as defective or non-defective with high accuracy. Trained on 7,348 industrial images. Features real-time web interface for instant analysis and confidence scoring.

Casting Product Defect Detection System
AI-powered defect detection system for manufacturing quality control. Uses deep learning (ResNet18) to classify metal casting products as defective or non-defective with high accuracy. Trained on 7,348 industrial images with real-time web interface for instant analysis.

Table of Contents
Overview
Features
Dataset
Model Architecture
Installation
Usage
Training
Results
Technologies Used
Project Structure
Future Improvements
License
Overview
This project implements an automated visual inspection system for detecting defects in metal casting products. The system uses transfer learning with a ResNet18 convolutional neural network to classify casting images as either defective or non-defective, helping manufacturers improve quality control efficiency.

Features
Binary Classification: Distinguishes between defective and non-defective casting products
High Accuracy: Achieves strong performance on validation data
Real-time Inference: Instant predictions through web interface
User-friendly Interface: Gradio-based web app requiring no technical knowledge
Confidence Scores: Provides probability breakdown for predictions
Transfer Learning: Leverages pre-trained ResNet18 for faster training and better performance
Dataset
Source: Casting Product Image Data for Quality Inspection

Total Images: 7,348 grayscale images
Image Size: Resized to 512x512 pixels
Classes:
def_front: Defective castings (surface cracks, porous areas, dimensional defects)
ok_front: Non-defective castings (smooth surfaces, proper quality)
Split: 80% training, 20% validation
Model Architecture
Base Model: ResNet18 (pre-trained on ImageNet)
Modification: Final fully connected layer replaced for binary classification
Input Size: 512x512x3 (RGB)
Output: 2 classes (Defective, Non-Defective)
Optimizer: Adam
Loss Function: Cross-Entropy Loss
Learning Rate: 0.001 with ReduceLROnPlateau scheduler
Installation
Requirements
bash
Python 3.8+
PyTorch 1.10+
torchvision
gradio
numpy
opencv-python
Pillow
matplotlib
tqdm
Setup
Clone the repository:
bash
git clone https://github.com/yourusername/casting-defect-detection.git
cd casting-defect-detection
Install dependencies:
bash
pip install torch torchvision gradio numpy opencv-python pillow matplotlib tqdm
Download the dataset from Kaggle and place it in the project directory
Usage
For Google Colab (Recommended)
Upload the training notebook to Google Colab
Mount Google Drive
Upload dataset to Drive
Run training cells
Run deployment cell to launch web app
Running the Web App
python
# Load the trained model and launch Gradio interface
python app.py
The app will generate a public URL that you can share for testing.

Making Predictions
Open the generated URL in your browser
Upload an image of a casting product
Click "Analyze Image"
View the prediction result and confidence scores
Training
Data Preprocessing
Resize images to 512x512
Normalize pixel values
Data augmentation:
Random horizontal flips
Random rotation (±10 degrees)
Color jitter (brightness, contrast)
Training Process
python
# Key hyperparameters
EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 0.001
Training takes approximately 20-40 minutes on GPU (Google Colab T4).

Model Saving
The best model (highest validation accuracy) is automatically saved to Google Drive as best_model.pth.

Results
Training Accuracy: ~95%+
Validation Accuracy: ~90%+
Inference Time: <1 second per image
Sample Predictions
The model successfully identifies:

Surface cracks and porous defects
Dimensional irregularities
Manufacturing flaws
Quality casting products
Technologies Used
Deep Learning Framework: PyTorch
Pre-trained Model: ResNet18 (ImageNet weights)
Web Framework: Gradio
Development Environment: Google Colab
Data Storage: Google Drive
Image Processing: OpenCV, Pillow
Visualization: Matplotlib
Project Structure
casting-defect-detection/
│
├── training_notebook.ipynb      # Model training code
├── deployment_app.py             # Gradio web application
├── best_model.pth               # Trained model weights
├── README.md                    # Project documentation
│
├── dataset/
│   ├── def_front/              # Defective casting images
│   └── ok_front/               # Non-defective casting images
│
└── requirements.txt            # Python dependencies
Future Improvements
 Multi-class defect classification (specific defect types)
 Object detection for defect localization
 Model optimization for edge deployment
 Integration with manufacturing execution systems (MES)
 Real-time video stream processing
 Deployment to production environments (Docker, AWS, etc.)
 Mobile application development
 Batch processing capabilities
 Extended dataset with more defect types
Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Acknowledgments
Dataset provided by Kaggle user ravirajsinh45
ResNet architecture from He et al. (2015)
Built with PyTorch and Gradio frameworks
Contact
For questions or feedback, please open an issue on GitHub.

Note: This project is for educational and research purposes. For production deployment in critical manufacturing environments, additional validation and testing is recommended.

