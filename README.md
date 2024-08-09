<h1>Prediction-of-Monkeypox-Disease</h1>
<hr><p>This project involves training a convolutional neural network (CNN) using TensorFlow and Keras to classify skin disease images. The dataset consists of labelled images of skin conditions, which are preprocessed, augmented, and used to train a CNN. The model's performance is evaluated using various metrics, including accuracy, loss, and confusion matrix analysis.</p><h2>General Information</h2>
<hr><ul>
<li>Data Preparation
Data Loading:</li>
</ul>
<p>Loaded preprocessed image data and corresponding labels from serialized files using Pickle.
Data Splitting:</p>
<p>Divided the dataset into training (80%), validation (10%), and testing (10%) sets using train_test_split from scikit-learn.
Normalization:</p>
<p>Scaled pixel values to the range [0, 1] to improve model performance.
Data Augmentation:</p>
<p>Applied image augmentation techniques using ImageDataGenerator to enhance model generalization. Techniques included rotation, zoom, width shift, and height shift.</p><ul>
<li>Model Building
Model Architecture:</li>
</ul>
<p>Built a Convolutional Neural Network (CNN) with depthwise separable convolutions using TensorFlow/Keras.
Architecture included:
Input Layer: 300x300 RGB images
Depthwise Separable Convolutional Layers: To extract features efficiently
Max Pooling Layers: To reduce dimensionality
Dropout Layer: To prevent overfitting
Flatten Layer: To convert feature maps to a 1D vector
Dense Output Layer: Softmax activation for classification into 4 categories
Model Compilation:</p>
<p>Optimizer: Adam
Loss Function: Sparse Categorical Crossentropy
Metrics: Accuracy
Model Training
Initial Training:</p>
<p>Trained the model using augmented training data with a batch size of 32 for 128 epochs.
Monitored validation accuracy and loss to prevent overfitting.
Final Training:</p>
<p>Refined the model with additional training using the same data augmentation techniques for 50 more epochs to achieve the best validation accuracy.
Model Evaluation
Training and Validation Performance:</p>
<p>Plotted training vs. validation accuracy and loss to visualize model performance.
Test Accuracy:</p>
<p>Evaluated model performance on the test set and reported accuracy.
Confusion Matrix:</p>
<p>Generated a confusion matrix to visualize the performance across different classes.
Plotted the confusion matrix to assess the accuracy of predictions and identify misclassifications.</p><h2>Technologies Used</h2>
<hr><ul>
<li><strong>Programming Languages</strong>: Python</li>
</ul><ul>
<li><strong>Libraries</strong>: TensorFlow, Keras, OpenCV, NumPy, Pandas, Matplotlib, Seaborn, scikit-learn</li>
</ul><ul>
<li><strong>Visualization</strong>: Matplotlib, Seaborn</li>
</ul><ul>
<li><strong>Data Storage</strong>: Pickle for data serialization</li>
</ul><ul>
<li><strong>Model Evaluation</strong>: Confusion Matrix, Classification Report</li>
</ul><ul>
<li><strong>Conclusion</strong></li>
</ul>
<p>In this study, we have presented the open-source “Monkeypox Skin Image Dataset (MSID)” for automatic detection of Monkeypox from skin disease and performed an initial study using deep learning models. Despite being a small dataset, the promising results (i.e. 71.43% using Depthwise Convolution) but more samples with a better geographical, racial, and gender distribution can enhance the results for practical AI use. We also believe that our web app prototype will assist the Monkeypox suspects in conducting preliminary screening from the comforts of home and enable them to take adequate action in the early stages of the infection</p><h2>Contact</h2>
<hr><p><span style="margin-right: 30px;"></span><a href="https://github.com/onkarrainak/onkarrainak/blob/main/www.linkedin.com/in/onkarrainak"><img target="_blank" src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" style="width: 10%;"></a><span style="margin-right: 30px;"></span><a href="https://github.com/OnkarRainak"><img target="_blank" src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" style="width: 10%;"></a></p>
