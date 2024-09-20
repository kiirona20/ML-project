import numpy as np                   # import numpy package under shorthand "np"
import pandas as pd                  # import pandas package under shorthand "pd"

from sklearn.model_selection import train_test_split 

df = pd.read_csv('mushrooms.csv')


#data preprosessing

#Remove unused features
df = df.drop(columns=['stalk-root','habitat', 'population','odor'], axis=1)
print(df.head(5))

# Convert class to binary data (1=poisonous 0=edible)
binaries = pd.get_dummies(df["class"], dtype=int)
print(binaries)
df_two = pd.concat((binaries, df), axis=1)
df_two = df_two.drop(['class'], axis=1)
df_two = df_two.drop(['e'], axis=1)

result = df_two.rename(columns={'p': 'class'})
print(result.head(5))

#Shuffle the data:
result = result.sample(frac = 1)
print(result.head(5))

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


#Split the data into training, validation and test data
#Ratio 70/15/15
# Split the dataset into a training set and a validation set

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, shuffle=False) #Already shuffled

# split the remaining 30% into 15% validation and 15% test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, shuffle=False)

#Should be train = 5686, val = 1219, test = 1219
print("training size:", len(X_train))
print("validation size:", len(X_val))
print("test size:", len(X_test))

