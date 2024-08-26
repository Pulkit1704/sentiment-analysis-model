import tensorflow as tf 
from data_utils import load_data
from rnn_model import RNNModel 

BATCH_SIZE = 256

train_set, test_set = load_data("./data/processed_data", 0.5) 

train_set = train_set.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_set = test_set.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE) 

model = RNNModel() 

# the forward pass is working, complete the model class with fit method and loss function and run on the train_set. 

model.compile(optimizer = tf.keras.optimizers.Adam(1e-4), loss = tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics = "accuracy") 

print("training the model...")
model.train_encoder(train_set)
model.fit(train_set, epochs = 10, validation_data=test_set, validation_steps= 10)