import tensorflow as tf 


class RNNModel(tf.keras.Model): 

    def __init__(self, vocabulary_size = 10_000): 
        super().__init__()
        self.vocab_size = vocabulary_size

        self.textEncoder = tf.keras.layers.TextVectorization(max_tokens=self.vocab_size) 

        self.embedding = tf.keras.layers.Embedding(input_dim = self.vocab_size + 1 , output_dim = 32, mask_zero = True) 
        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units = 32))
        self.dense1 = tf.keras.layers.Dense(units = 32)
        self.dense2 = tf.keras.layers.Dense(units = 1) 

        self.activation = tf.keras.layers.ReLU() 
        


    def call(self, x): 
        x = self.textEncoder(x)

        x = self.lstm(self.embedding(x)) 

        output = self.dense2(self.activation(self.dense1(x))) 

        return self.activation(output) 



    def train_encoder(self, train_data: tf.data.Dataset):

        self.textEncoder.adapt(train_data.map(lambda text, labels: text)) 
    
