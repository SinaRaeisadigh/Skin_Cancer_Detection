# Skin_Cancer_Detection

This notebook is intended for skin cancer detection using the HAM10000 dataset, which involves using image processing, data augmentation, and training a convolutional neural network (CNN) based on MobileNet. Here’s a detailed breakdown:

## 1. Import Libraries
The necessary libraries are imported for handling image processing, visualization, data manipulation, machine learning, and deep learning. Some key libraries include:

PIL for image processing.
pandas, numpy for data handling and numerical calculations.
TensorFlow/Keras for building and training the deep learning model.
seaborn, matplotlib for visualizations.
sklearn for evaluation metrics.

## 2. Data Preprocessing
This section prepares the data for analysis:

Loading Metadata: The HAM10000 metadata is loaded into a DataFrame (df) using pd.read_csv.
Missing Values Handling: The missing values in the age column are replaced with the mean age. Other null values are addressed to ensure data consistency.
Lesion Type Mapping: The labels in the dataset (dx) are mapped to more descriptive names using a dictionary.
Image Paths & Loading: The paths of images are constructed from the given directories, and the images are loaded into the DataFrame by resizing each to (125, 100). This step ensures that all images are of uniform size.

## 3. Sample Visualization
This section provides a visualization of a few samples of each skin lesion type.

Plotting Image Samples: A grid plot (7x5) shows five random samples of each lesion type, helping to visualize the characteristics of each type.
Exploratory Data Analysis (EDA): Additional visualizations include bar plots of the distribution of age, gender, localization, and cell types in the dataset. These visualizations provide insight into the demographic breakdown of the patients and the lesion types.

## 4. Splitting Data for Training and Testing
Feature and Target Definition: Features (df.drop(columns=['cell_type_idx'])) and the target (df['cell_type_idx']) are defined.
Splitting into Train/Test Sets: The data is split into train and test sets (75%-25%). Later, a validation set (10%) is derived from the training set.
Standardization and Normalization: The image data is normalized using mean and standard deviation to improve model convergence.

## 5. Model Building
MobileNet Model Adaptation: The pre-trained MobileNet model is used as the base model.
The change_model function adapts the input shape to fit the dataset’s image size. The base model is modified to have the desired input shape and some layers are excluded to create a custom architecture.
New Layers Added:
GlobalAveragePooling2D is applied to reduce the spatial dimensions.
Dropout is added to prevent overfitting.
A new Dense layer with 7 output classes and softmax activation is added for the final classification of skin lesion types.
Freezing Layers: The layers of MobileNet are frozen except for the last 23 layers to ensure that only the necessary parts are trained.
Compilation: The model is compiled with the Adam optimizer, categorical_crossentropy as the loss function, and evaluation metrics like categorical_accuracy, top-2 accuracy, and top-3 accuracy.

## 6. Data Augmentation
Image Augmentation: To enrich the training dataset and reduce overfitting, data augmentation is applied to the training images. The augmentation includes:
Random rotations, zoom, and shifts in width/height.
Horizontal and vertical flips.
The augmented data is generated dynamically during training to provide the model with diverse images.

## 7. Model Training
Callbacks for Training:
ModelCheckpoint is used to save the model that has the best top-3 accuracy on the validation set.
ReduceLROnPlateau reduces the learning rate when a metric (val_top_3_accuracy) has stopped improving.
Training:
The model is trained using the augmented images with fit().
The training process also uses class weights to make the model more sensitive to underrepresented classes, particularly melanoma, which is given a higher weight to address the class imbalance.

## 8. Model Evaluation
Evaluating the Model:
The trained model is evaluated on the test dataset.
Metrics include validation loss, categorical accuracy, top-2 accuracy, and top-3 accuracy to assess the model’s performance on unseen data.
