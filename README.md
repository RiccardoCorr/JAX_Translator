# JAX_Translator
To achieve the goal of creating a translator I decided to implement a sequence-to-sequence model with an Encoder-Decoder architecture, each powered by a single-layer
 LSTM (Long Short-Term Memory) cell, implemented from scratch in JAX.

The model operates in two distinct workflows: one for training and another for generation, which will be implemented and explained in the second exercise.

**Training workflow**:
During training, the model learns to translate source sequences to target sequences through the following steps:

1. The *Encoder* processes the input sequence:

   - Converts tokens into embeddings.
   - Passes embeddings through the LSTM.
  - Outputs the final hidden and cell states.

2. The *Decoder* receives Encoder's final states.

3. At each timestep the *Decoder*:

   - Takes the ground truth token from the target sequence as input.
   - Processes it through the LSTM, updating its states.
   - Projects LSTM output to vocabulary space.
   - Outputs logits for loss computation.

**Generation workflow**:
1. The input sequence (a tokenized and padded English sentence) is encoded using the same process as during training, producing the encoder's hidden and cell states.

2. The Decoder starts with the start token (<SOS>) as the initial input.

3. At each timestep, until *'max_length'* is reached, the following steps are repeated:

   - The current token is embedded using the embedding matrix.
   - The embedded vector is passed through the LSTM, updating the hidden and cell states.
   - The updated hidden state is projected into the vocabulary space, producing logits.
   - The next token is determined by applying an argmax operation on the logits (deterministic selection).
   - The selected token is fed back into the decoder as input for the next timestep in an autoregressive way.


 4. The complete generated sentence is returned as output.
