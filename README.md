# spam_Detection_mL-model-using-svm
### Project README.md

Here is a README file description for your End-to-End SMS/Email Spam Detection project using Naive Bayes, based on the uploaded Jupyter Notebook file.

#### **Project Title**
End-to-End Project on SMS/Email Spam Detection using Naive Bayes

---

#### **Overview**

This project demonstrates a complete workflow for building an SMS/email spam detection model using the Multinomial Naive Bayes algorithm. The process includes data preprocessing, exploratory data analysis, feature engineering with `CountVectorizer`, and model training and evaluation.

---

#### **Dataset**

The project utilizes a dataset containing SMS messages and their corresponding categories, either "ham" (legitimate) or "spam". The dataset has 5572 rows and two columns: `Category` and `Message`.

---

#### **Methodology**

1.  **Data Loading and Exploration**: The project starts by loading the dataset and performing an initial check on the data, including value counts for spam and ham messages. The dataset has 4825 'ham' and 747 'spam' entries.
2.  **Data Splitting**: The data is split into training and testing sets to evaluate the model's performance on unseen data.
3.  **Feature Engineering**: The text data is transformed into a numerical format using `CountVectorizer` from the Scikit-learn library. This technique converts the text messages into a matrix of token counts.
4.  **Model Training**: A `Multinomial Naive Bayes` classifier is trained on the vectorized training data.
5.  **Model Evaluation**: The trained model's performance is evaluated on the test set. The model accuracy is calculated using `model.score(X_test_cv, y_test)`.

---

#### **Model Accuracy**

The accuracy of the trained model on the test data is 98.7%.

---

#### **Requirements**

The project uses the following Python libraries:
* pandas
* numpy
* scikit-learn
    * `CountVectorizer`
    * `train_test_split`
    * `MultinomialNB`
* `matplotlib.pyplot` (for visualization)
* `seaborn` (for visualization)

---

#### **How to Run the Code**

1.  Clone the repository from GitHub.
2.  Ensure you have the required libraries installed.
3.  Open and run the `Spam_Detection.ipynb` Jupyter Notebook.
4.  The notebook will guide you through each step of the process, from data loading to model evaluation.
