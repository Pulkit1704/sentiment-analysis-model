import tensorflow as tf 


class RNNModel(tf.keras.Model): 

    def __init__(self, vocabulary_size: int = 1000, **kwargs): 
        super().__init__(**kwargs)
        self.vocab_size = vocabulary_size

        self.embedding = tf.keras.layers.Embedding(input_dim = self.vocab_size + 1 , output_dim = 64, mask_zero = True) 
        self.lstm = tf.keras.layers.LSTM(units = 64) 
        self.bidirectional = tf.keras.layers.Bidirectional(self.lstm)
        self.dense1 = tf.keras.layers.Dense(units = 32, activation = 'relu')
        self.dense2 = tf.keras.layers.Dense(units = 1) 
        self.dropout = tf.keras.layers.Dropout(0.8) 

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embedding_config": self.embedding.get_config(),
            "lstm_config": self.lstm.get_config(),
            "dense1_config": self.dense1.get_config(),
            "dense2_config": self.dense2.get_config(),
            "dropout_config": self.dropout.get_config()
        })
        return config
    
    @classmethod
    def from_config(cls, config: dict):
        # Extract layer configurations
        vocab_size = config.pop("vocab_size") 
        embedding_config = config.pop("embedding_config")
        lstm_config = config.pop("lstm_config")
        dense1_config = config.pop("dense1_config")
        dense2_config = config.pop("dense2_config")
        dropout_config = config.pop("dropout_config") 

        # Create a new instance of the model
        instance = cls(**config)

        # Rebuild the layers with the saved configurations
        instance.vocab_size = vocab_size
        instance.embedding = tf.keras.layers.Embedding.from_config(embedding_config)
        instance.lstm = tf.keras.layers.LSTM.from_config(lstm_config)
        instance.bidirectional = tf.keras.layers.Bidirectional(instance.lstm) 
        instance.dense1 = tf.keras.layers.Dense.from_config(dense1_config)
        instance.dense2 = tf.keras.layers.Dense.from_config(dense2_config)
        instance.dropout = tf.keras.layers.Dropout.from_config(dropout_config) 

        return instance

    def call(self, x: tf.Tensor): 

        x = self.bidirectional(self.embedding(x))

        output = self.dense2(self.dropout(self.dense1(x))) 

        return output 
