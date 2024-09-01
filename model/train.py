import tensorflow as tf 
from data_utils import load_data
from rnn_model import RNNModel 
import matplotlib.pyplot as plt 

BATCH_SIZE = 256
VOCAB_SIZE = 500
EPOCHS = 100

train_set, test_set = load_data("./data/processed_data", 0.7) 

train_set = train_set.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE) 
test_set = test_set.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE) 

model = RNNModel(VOCAB_SIZE) 

model.compile(optimizer = tf.keras.optimizers.Adam(0.001), loss = tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics = "accuracy") 

model.train_encoder(train_set)
history = model.fit(train_set, epochs = EPOCHS)

model.save_weights("./model/trained_model/")

plt.plot(range(1, EPOCHS + 1), history.history["loss"], marker = "o")
plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.savefig("./model/plots/training_loss.png") 
plt.close() 

plt.plot(range(1, EPOCHS + 1), history.history["accuracy"], marker = "o", color = "red")
plt.xlabel("Epochs") 
plt.ylabel("Training accuracy") 
plt.savefig("./model/plots/training_accuracy.png")
plt.close() 