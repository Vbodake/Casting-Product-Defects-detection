# Casting-Product-Defects-detection
AI-powered defect detection system for manufacturing quality control. Uses deep learning (ResNet18) to classify metal casting products as defective or non-defective with high accuracy. Trained on 7,348 industrial images. Features real-time web interface for instant analysis and confidence scoring.

Dataset Link: https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product


- Total Images: 7,348 grayscale images

- Image Size: Resized to 512x512 pixels

Classes:

- def_front: Defective castings (surface cracks, porous areas, dimensional defects)

- ok_front: Non-defective castings (smooth surfaces, proper quality)



Split: 80% training, 20% validation


## setup

1. Clone the repository:

2. Install dependencies: "pip install torch torchvision gradio numpy opencv-python pillow matplotlib tqdm"

3. Download the dataset from Kaggle and place it in the project directory

## Usage

For Google Colab 

1. Upload the training notebook to Google Colab
2. Mount Google Drive
3. Upload dataset to Drive
4. Run training cells
5. Run deployment cell to launch web app

## Future Improvements

 - Multi-class defect classification (specific defect types)
 - Object detection for defect localization
 - Model optimization for edge deployment
 - Integration with manufacturing execution systems (MES)
 - Real-time video stream processing
 - Deployment to production environments (Docker, AWS, etc.)
 - Mobile application development
 - Batch processing capabilities
 - Extended dataset with more defect types


### Contributions are welcome! Please feel free to submit a Pull Request.

##  Acknowledgments

- Dataset provided by Kaggle user ravirajsinh45
- ResNet architecture from He et al. (2015)
- Built with PyTorch and Gradio frameworks
