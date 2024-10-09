import numpy as np                   # import numpy package under shorthand "np"
import pandas as pd                  # import pandas package under shorthand "pd"
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier  # Import Random Forest
from sklearn.metrics import accuracy_score, confusion_matrix  # evaluation metrics
from sklearn.metrics import log_loss


df = pd.read_csv('mushrooms.csv')


#data preprosessing

#Remove unused features
df = df.drop(columns=['stalk-root','habitat', 'population','odor'], axis=1)
#print(df.head(5))

# Convert class to binary data (1=poisonous 0=edible)
binaries = pd.get_dummies(df["class"], dtype=int)
#print(binaries)
df_two = pd.concat((binaries, df), axis=1)
df_two = df_two.drop(['class'], axis=1)
df_two = df_two.drop(['e'], axis=1)

result = df_two.rename(columns={'p': 'class'})
#print(result.head(5))

#Shuffle the data:
result = result.sample(frac = 1)
#print(result.head(5))

#Some data analysis
# number of entries
num_data_points = len(result)
print("number of data points: ", num_data_points)
# count the occurrences of each class
class_counts = result['class'].value_counts()
print(class_counts)
# calculate percentage the poisonous mushrooms
poisonous_percentage = (class_counts[1]/len(result)) * 100

print("percentage of poisonous mushrooms: ", poisonous_percentage)
print("percentage of edible mushrooms: ", 100-poisonous_percentage)


X = result.drop(columns=['class'])  # All features
y = result['class']  # The label 
print(X.head(5))

#Convert X data to be numerical so the regressions will work.

X = pd.get_dummies(X, dtype=int)

#Split the data into training, validation and test data
#Ratio 70/15/15
# Split the dataset into a training set and a validation set

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, shuffle=True) #Already shuffled

# split the remaining 30% into 15% validation and 15% test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, shuffle=True)


#Logistic regression

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)  
lr_y_pred_proba = clf.predict_proba(X_test)

# Validate the model
val_y_pred = clf.predict(X_val)
val_acc = accuracy_score(y_val, val_y_pred)
print(f'Validation Accuracy: {val_acc}')

# Random Forest
rf_clf = RandomForestClassifier(random_state=0)
rf_clf.fit(X_train, y_train)
rf_y_pred = rf_clf.predict(X_test) 
rf_y_pred_proba = rf_clf.predict_proba(X_test) 


# Validate the model
val_y_pred_rf = rf_clf.predict(X_val)
val_acc_rf = accuracy_score(y_val, val_y_pred_rf)
print(f'Validation Accuracy forest: {val_acc_rf}')

# Compare accuracy
rf_acc = accuracy_score(y_test, rf_y_pred)
lr_acc = accuracy_score(y_test, y_pred)
print("Accuracy Random forest: ", rf_acc)
print("Accuracy logistic regression: ", lr_acc)


# Compare Log Loss
rf_logloss = log_loss(y_test, rf_y_pred_proba)
lr_logloss = log_loss(y_test, lr_y_pred_proba)

print("Random forest Log Loss: ", rf_logloss)
print("Logistic regression Log loss: ", lr_logloss)



# Logistic Regression confusion matrix
confmat = confusion_matrix(y_test, y_pred)
#Random tree condusion matrix
rf_confmat = confusion_matrix(y_test, rf_y_pred)


# Plot the confusion matricies
ax = plt.subplot()
sns.heatmap(confmat,annot=True, fmt='g', ax=ax)
ax.set_xlabel('Predicted labels',fontsize=15)
ax.set_ylabel('True labels',fontsize=15)
plt.show()


ax = plt.subplot()
sns.heatmap(rf_confmat, annot=True, fmt='g', ax=ax)
ax.set_xlabel('Predicted labels', fontsize=15)
ax.set_ylabel('True labels', fontsize=15)
plt.show()

















