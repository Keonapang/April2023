# Breast cancer detection (ANN)

## If using google colab, don't miss this step!
```
from google.colab import drive
drive.mount('/content/drive')
```

## Import important libraries (keras and scikit-learn) to begin this project
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import pylab as pl
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten 
from keras.layers import Activation
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import recall_score 
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
```
## Loading data from the Breast Cancer Classification .csv file
```
data_raw = pd.read_csv('/content/drive/MyDrive/Colab-Notebooks/BCW_dataset.csv', delimiter=',', header=0, index_col=None) # Head method show first 5 rows of data
```

## Data Cleaning

### Drop unused columns in pandas dataframe. Here, we dropped 3 columns
```
drop_columns = ['Unnamed: 32', 'diagnosis'] #a list of strings (column headers)
```
### Convert Strings ('M' or 'B') -> Integers ('1' or '0')
```
d = {'M': 0, 'B': 1} #for mapping 
```

### Define features and labels: map Malignant('M') = 0 and Benign ('B') = 1 to each letter in the 'diagnosis column'
```
y = data_raw['diagnosis'].map(d) #map 'M' as 0 and 'B' as 1
```

### Drop columns from our pandas dataframe, now instead of 33 columns, we have 31 columns.
```
X = data_raw.drop(drop_columns, axis=1) #drop_columns dictates which columns are gone
```
## Defining variables for the For loop and model building
```
columns = list(X) # Get list of column headers from a Pandas DataFrame
lon = []
complete = 30
num_input_list = [30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]
num_hidden_nodes = 20 #can change 
num_output = 2
num_input = 30
list_accuracy = []
list_loss = []
```

## For loop 

```
for n in range(30): #incrementally drops column headers from list(X), till there is only one column header left 
  lon.append(n)
  n+1
  X_drop = X.drop(X.columns[lon], axis=1) #X data in a Pandas DataFrame
  #scale the input data X_drop
  scaler = StandardScaler() 
  X_scaled = scaler.fit_transform(X_drop)
  encode_array = OneHotEncoder()
  y_2OP = encode_array.fit_transform(y[:, None]).toarray()
  #Split the dataset into training (75%) and test (25%) 
  X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_2OP, test_size=0.25, random_state=0)
  
  def model_1():
    # create model
    model = Sequential()
    model.add(Dense(num_hidden_nodes, input_dim= (complete - n), activation='relu')) #Dense : a regular fully connected layer
    model.add(Dense(num_output, activation='softmax')) #output layer
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Choice of optimizer: adam (adaptive moment estimation), AdaGrad (adaptive learning rate), 
    # sgd (Stochastic gradient descent), RMSprop (similar to AdaGrad), Adadelta (adaptive delta) ...
    return model

  model = model_1() #define model name
  print("Number of features in the model below: ", complete-n)
  history = model.fit(X_train, y_train, batch_size=8, epochs=60,verbose=2, validation_data=(X_test, y_test))

  score = model.evaluate(X_test, y_test, verbose=0)
  list_accuracy.append(score[1])
  list_loss.append(score[0])
```

## ANN performance metrics (test loss and test accuracy) are stored in a list of integers. 
```
print('Test loss:', list_loss[0:10])
print('Test accuracy:', list_accuracy[0:10])
```
## Plot a graph describing the relationship between the Number of features (x-axis) and Test accuracy (y-axis).
```
x_features = [30, 29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]
plt.plot(x_features, list_accuracy)
plt.title("Number of features (x-axis) vs Test accuracy (y-axis)")
plt.ylabel('Test accuracy', fontsize = 12)
plt.xlabel('Number of features', fontsize = 12)
plt.show()
```

