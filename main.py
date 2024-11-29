import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf



data = pd.read_csv('emotions_5.csv')
print(data.head())


#taking all rows and all cols without last col for X which include features
#taking last col for Y, which include the emotions
X = data.iloc[: ,:-1].values
Y = data['Emotions'].values

#One-Hot Encoding the labels

onehot_encoder = OneHotEncoder()
Y = onehot_encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

print(f"Features shape: {X.shape}, target shape : {Y.shape}")


# train test split
X_train , X_test , y_train , y_test = train_test_split(X , Y , test_size= 0.2 , random_state=123)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)



# stander scaling for the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=10,
                                                     restore_best_weights=True)

lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.5,min_lr=0.00001)

# Build a simple feedforward neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, input_dim=X_train.shape[1] , activation='relu', kernel_initializer="uniform"),
    tf.keras.layers.Dropout(0.5),  # Dropout for regularization
    tf.keras.layers.Dense(64, activation='relu', kernel_initializer="uniform"),
    tf.keras.layers.Dropout(0.5),  # Dropout for regularization 
    tf.keras.layers.Dense(32, activation='relu', kernel_initializer="uniform"),
    tf.keras.layers.Dropout(0.5),  # Dropout for regularization 
    tf.keras.layers.Dense(y_train.shape[1] , activation='softmax')  # Output layer for classification
])



# Compile the model with an appropriate loss function and optimizer
model.compile(optimizer= 'adam' , loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()


# Train the model
history = model.fit(X_train, y_train,
                     epochs=100, batch_size=64,
                       validation_data=(X_test, y_test),
                       callbacks=[early_stopping],
                       verbose = 1
                       )

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}")


# saving the model

import joblib as jb
jb.dump(model,'model_2.pkl')


import matplotlib.pyplot as plt

# Plot training & validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Plot training & validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d')
plt.show()

print(classification_report(y_test, y_pred_classes))
