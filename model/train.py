import tensorflow as tf 
from data_utils import load_data
from rnn_model import RNNModel 
import matplotlib.pyplot as plt 

BATCH_SIZE = 128
VOCAB_SIZE = 300
EPOCHS = 50

def save_history(history, filepath): 

    for key, value in history.items():

        plt.plot(range(1, len(value)+1), value, marker = "o") 
        plt.ylabel(f"{key}")
        plt.xlabel("Epochs")
        plt.grid(True) 
        plt.savefig(f"{filepath}/{key}.png") 
        plt.close()


train_set, test_set = load_data("./data/processed_data", 0.7) 

train_set = train_set.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE) 
test_set = test_set.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE) 

model = RNNModel(VOCAB_SIZE) 

model.compile(optimizer = tf.keras.optimizers.Adam(0.01), loss = tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics = ["accuracy"]) 

callback = tf.keras.callbacks.EarlyStopping(monitor = "val_accuracy", patience = 5, restore_best_weights= True)

history = model.fit(train_set, epochs = EPOCHS, validation_data = test_set, callbacks = [callback])

model.summary()

model.save("./trained_model/model.keras") 

save_history(history.history, "./trained_model/training_plots/")