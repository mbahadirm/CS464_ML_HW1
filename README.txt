#Description
This program implements a Naïve Bayes classifier to classify astronomical objects based on features extracted from space observations. The dataset contains measurements such as redshift, infrared filter, and ultraviolet filter values. Additionally, a Hybrid Naïve Bayes model is trained, which considers dependencies among selected features. The program also ranks features based on mutual information and evaluates the effect of selecting top-k features on model performance.

#Requirements
Ensure you have the following Python packages installed:

-pandas
-numpy
-matplotlib.pyplot
You can install them using:

pip install pandas numpy matplotlib.pyplot

#Usage
To run the program, navigate to the directory where the script is located and execute the following command:
python main.py
Make sure X_train.csv, y_train.csv, X_test.csv, and y_test.csv are in the same directory as the script.
Also you can directly run using vs code run without debugging function

#Output
The program prints:

-Model accuracy for both Naïve Bayes and Hybrid Naïve Bayes classifiers.
-Confusion matrix for performance evaluation.
-The top 3 feature values contributing to classifying an object as a galaxy or otherwise.
-Ranked features based on their mutual information with the class variable.
-Model performance as top-k features are selected.