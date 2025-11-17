This project implements a character-level text generator using a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) units. The model is trained on a small text corpus and learns to predict the next character given a sequence of previous characters. After training, it can generate new text in a similar style to the input.

Project Description The goal of this project is to explore sequence modeling using LSTMs by training a character-level text generator. The project follows the required steps:

Select a text corpus

Build a character-level RNN/LSTM model

Train the model to predict the next character

Sample from the trained model to generate new text The implementation uses TensorFlow/Keras and trains on a small Shakespeare-like sample corpus.

Architecture Explanation Data Processing The corpus is converted to lowercase. A character vocabulary is built from all unique characters. Input sequences of length 50 characters are created. The target for each sequence is the next character.

Model Architecture The neural network consists of: Embedding Layer Converts integer-encoded characters into dense vector embeddings. LSTM Layer (512 units) Learns temporal dependencies in the sequence data. Dense Output Layer (Softmax) Outputs a probability distribution over the vocabulary for the next character.

Training Loss: Categorical Crossentropy Optimizer: Adam Epochs: 5 (demo level) ModelCheckpoint is used to save the best model weights.

Text Generation A seed sequence initializes text generation. Predictions are sampled using temperature scaling for controllable creativity: Low temperature → more predictable text High temperature → more diverse text

How to Run the Code Install needed libraries: pip install tensorflow numpy
Run the script python RNN.py

What happens when you run it: Preprocesses text and builds training sequences Trains the LSTM model Generates text using two temperature values Prints outputs directly to the terminal

Model weights are saved to: model_weights.h5

Results and Analysis Training Performance Given the tiny demo corpus, the model learns only basic character transitions, not full language structure. Loss decreases quickly but does not converge to strong language fluency.
Generated Text Two-generation modes illustrate the effect of sampling temperature: Temperature = 0.2: Produces predictable and repetitive patterns. Output tends to mimic the training text structure closely.

Temperature = 1.0: Produces more varied, creative, and chaotic text. Demonstrates the model’s understanding of basic character distributions.

Limitations Tiny dataset limits model complexity. Character-level modeling requires longer training for coherent results. Increasing corpus size and epochs would significantly improve performance.

Team Member Contribution Anna: Implemented data processing pipeline and sequence generation logic Kripamoye: Built and tuned the LSTM architecture Fabiola: Wrote text generation sampling function and testing utilities Roselio: Prepared README, cleaned code, and performed final evaluation