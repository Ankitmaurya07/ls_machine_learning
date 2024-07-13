import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Rescaling
from tensorflow.keras import Input


dataset = tf.keras.preprocessing.image_dataset_from_directory(
    r'C:\Users\pashu\OneDrive\Desktop\homer_bart',
    image_size=(64, 64),
    label_mode="binary"
)


train_data = dataset.take(8)
test_data = dataset.skip(8)


train_data = train_data.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_data = test_data.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)


preprocess = tf.keras.Sequential([
    Rescaling(1./255)  # rescaling factor
])


model = tf.keras.Sequential()
model.add(Input((64, 64, 3)))
model.add(preprocess)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Correct loss function name
              metrics=['accuracy'])


model.fit(
    train_data,
    epochs=50,
    batch_size=64,
    verbose=1,
    validation_data=test_data
)


test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test accuracy: {test_accuracy}")
