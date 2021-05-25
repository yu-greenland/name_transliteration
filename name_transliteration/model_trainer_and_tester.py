import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import time
import datetime
import pickle
from collections import OrderedDict
from tensorflow.keras import backend as K

class ModelTrainerAndTester():
    """
    The purpose of the ModelTrainerAndTester class is to train a model (using training data produced from cleansing class) and evaluate that model (using testing data produced from cleansing class).
    This class is a modified version of https://keras.io/examples/nlp/lstm_seq2seq/.
    Primarily uses the Keras library to achieve this.

    Represents a model trainer and tester class, part of the overall name transliteration training pipeline.
    """
    
    def __init__(self, batch_size=32, epochs=10, latent_dim = 256, train_data_path = None, test_data_path = None, language="unknown", num_samples = None):
        """
        Edit variables on start up according to whether it is going to be a new model trainer and tester or loading from a previous run.
        """
        self.batch_size = batch_size
        self.epochs = epochs
        self.latent_dim = latent_dim
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.language = language
        self.num_samples = num_samples

        # these are defined later on

        self.num_encoder_tokens = 0
        self.num_decoder_tokens = 0
        self.model = None
        # self.encoder_input_data = None
        # self.decoder_input_data = None
        # self.decoder_target_data = None
        # this is a dictionary mapping input tokens (eg. english letters) to an index number
        self.input_token_index = None
        # this is a dictionary mapping target tokens (eg. japanese characters) to an index number
        self.target_token_index = None
        self.encoder_model = None
        self.decoder_model = None
        self.reverse_target_char_index = None
        self.max_decoder_seq_length = 0
        self.max_encoder_seq_length = 0
        self.history = None
        self.history_losses = None

        # we are going to use an ordered dictionary to represent an ordered set
        # the keys are the characters and the value is always going to be None
        self.input_characters = OrderedDict()
        self.target_characters = OrderedDict()

        # to time training runs
        self.elapsed_time = None

        self.results1 = None
        self.results2 = None
        self.results3 = None

    
    def processData(self, data_path):
        """
        Processes the text file data into a format that Keras can use to build seq2seq model.
        This should only be called after the determineDimensions() method.
        Processing invovles turning the input and target text files into a one hot vector encoding so that the Keras ML models can understand it.

        Parameters
        ----------
        data_path : str
            the data path to the text file containing tab seperated name pairs

        Returns
        ----------
        encoder_input_data : numpy array
            the input data turned into a one-hot-vector
        decoder_input_data : numpy array
            the target data turned into a one-hot-vector
        decoder_target_data : numpy array
            the target data turned into a one-hot-vector, but shifted forward by one time-step 
        """
        # Vectorize the data.
        input_texts = []
        target_texts = []

        with open(data_path, "r", encoding="utf-8") as f:
            lines = f.read().split("\n")
        if self.num_samples is None:
            self.num_samples = 9999999
        for line in lines[: min(self.num_samples, len(lines) - 1)]:
            input_text, target_text = line.split("\t")
            # We use "tab" as the "start sequence" character
            # for the targets, and "\n" as "end sequence" character.
            target_text = "\t" + target_text + "\n"
            input_texts.append(input_text)
            target_texts.append(target_text)
            # for char in input_text:
            #     if char not in input_characters:
            #         input_characters.add(char)
            # for char in target_text:
            #     if char not in target_characters:
            #         target_characters.add(char)

        # current_num_encoder_tokens = len(input_characters)
        # current_num_decoder_tokens = len(target_characters)
        # current_max_encoder_seq_length = max([len(txt) for txt in input_texts])
        # current_max_decoder_seq_length = max([len(txt) for txt in target_texts])

        # if self.num_encoder_tokens < current_num_encoder_tokens:
        #     self.num_encoder_tokens = current_num_encoder_tokens
        # if self.num_decoder_tokens < current_num_decoder_tokens:
        #     self.num_decoder_tokens = current_num_decoder_tokens
        # if self.max_encoder_seq_length < current_max_encoder_seq_length:
        #     self.max_encoder_seq_length = current_max_encoder_seq_length
        # if self.max_decoder_seq_length < current_max_decoder_seq_length:
        #     self.max_decoder_seq_length = current_max_decoder_seq_length

        # print("Number of samples:", len(input_texts))
        # print("Number of unique input tokens:", self.num_encoder_tokens)
        # print("Number of unique output tokens:", self.num_decoder_tokens)
        # print("Max sequence length for inputs:", self.max_encoder_seq_length)
        # print("Max sequence length for outputs:", self.max_decoder_seq_length)

        self.input_token_index = dict([(char, i) for i, char in enumerate(self.input_characters.keys())])
        # self.target_token_index = dict([(char, i) for i, char in enumerate(self.target_characters)])
        self.target_token_index = {}
        for i, char in enumerate(self.target_characters.keys()):
            self.target_token_index[char] = i

        # encoder_input_data is a one hot vector encoding of the input name
        # first dimension is the number of names, second dimension is the time step, third dimension is the 
        encoder_input_data = np.zeros(
            (len(input_texts), self.max_encoder_seq_length, self.num_encoder_tokens), dtype="float32"
        )
        decoder_input_data = np.zeros(
            (len(input_texts), self.max_decoder_seq_length, self.num_decoder_tokens), dtype="float32"
        )
        decoder_target_data = np.zeros(
            (len(input_texts), self.max_decoder_seq_length, self.num_decoder_tokens), dtype="float32"
        )

        # pretty sure this is building one hot vector representation of the data
        for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
            for t, char in enumerate(input_text):
                encoder_input_data[i, t, self.input_token_index[char]] = 1.0
            encoder_input_data[i, t + 1 :, self.input_token_index[" "]] = 1.0
            for t, char in enumerate(target_text):
                if char in self.target_token_index:
                    # decoder_target_data is ahead of decoder_input_data by one timestep
                    decoder_input_data[i, t, self.target_token_index[char]] = 1.0
                    if t > 0:
                        # decoder_target_data will be ahead by one timestep
                        # and will not include the start character.
                        decoder_target_data[i, t - 1, self.target_token_index[char]] = 1.0
            # since the input ended, we have to fill the rest of the max decoder seq length with spaces
            decoder_input_data[i, t + 1 :, self.target_token_index[" "]] = 1.0
            decoder_target_data[i, t:, self.target_token_index[" "]] = 1.0
        
        return encoder_input_data, decoder_input_data, decoder_target_data
    
    def buildModel(self):
        """
        Defines and creates the Keras model that is to train the name pair predictor.
        Is an encoder, decoder architecture using LSTM.
        Final layer is a dense layer with softmax activation.
        Built model is saved as the class level variable called "model"
        """
        # Define an input sequence and process it.
        encoder_inputs = keras.Input(shape=(None, self.num_encoder_tokens))
        encoder = keras.layers.LSTM(self.latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)

        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = keras.Input(shape=(None, self.num_decoder_tokens))

        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = keras.layers.LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        # softmax changes entire vector to a probability, all elements add to 1
        decoder_dense = keras.layers.Dense(self.num_decoder_tokens, activation="softmax")
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        self.model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    def trainModel(self, model_name, train_encode_input, train_decode_input, train_decode_output):
        """
        Trains the model and saves the trained model weights to disk

        Parameters
        ----------
        model_name : str
            the name of the folder of where the model weights exist
        train_encode_input : numpy array
            part of the input data to the model, should be a one-hot-vector encoding of input text
        train_decode_input : numpy array
            part of the input data of the model, should be a one-hot-vector encoding of the target text, but time shifted backwards by one step
        train_decode_output : numpy array
            the y component of the model, should be a one-hot-vector encording of the target text
        """
        # to stop the model when loss does not improve over some
        early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=4)

        self.model.compile(
            optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        # set shuffle seed tf.random.set_seed
        # use tensorflow data instead also call shuffle method, set parameter true
        # set model.fit steps - setting how many batches are run per epoch
        start_time = time.time()
        self.history = self.model.fit(
            x=[train_encode_input, train_decode_input],
            y=train_decode_output,
            validation_split=0.2,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=[early_stopping]
        )
        end_time = time.time()
        self.elapsed_time = end_time - start_time

        # Save model
        self.model.save(model_name)
    
    def createDecoderEncoder(self, model_name:str):
        """
        Loads a saved model and extracts the encoder and decoder layers from the model. Creates separate encoder and decoder models.
        Also creates a reverse look up table in the form of a dictionary to make sense of the predictions coming out of the decoder.

        Parameters
        ----------
        model_name : str
            the name of the saved model that is to be loaded
        """
        # Define sampling models
        # Restore the model and construct the encoder and decoder.
        self.model = keras.models.load_model(model_name)

        encoder_inputs = self.model.input[0]  # input_1
        encoder_outputs, state_h_enc, state_c_enc = self.model.layers[2].output  # lstm_1
        encoder_states = [state_h_enc, state_c_enc]
        self.encoder_model = keras.Model(encoder_inputs, encoder_states)

        decoder_inputs = self.model.input[1]  # input_2
        decoder_state_input_h = keras.Input(shape=(self.latent_dim,), name="input_3"+model_name)
        decoder_state_input_c = keras.Input(shape=(self.latent_dim,), name="input_4"+model_name)
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_lstm = self.model.layers[3]
        decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs
        )
        decoder_states = [state_h_dec, state_c_dec]
        decoder_dense = self.model.layers[4]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = keras.Model(
            [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
        )

        # Reverse-lookup token index to decode sequences back to
        # something readable.
        reverse_input_char_index = dict((i, char) for char, i in self.input_token_index.items())
        # self.reverse_target_char_index = dict((i, char) for char, i in self.target_token_index.items())
        self.reverse_target_char_index = {}
        for char, i in self.target_token_index.items():
            self.reverse_target_char_index[i] = char

    def decode_sequence(self, input_seq):
        """
        Performs inference on an input sequence. 
        This input sequence should be in the form of a one-hot-vector.
        Uses the created encoder and decoder model to iteratively predict the next character.

        Parameters
        ----------
        input_seq : numpy array
            the input encoded as a one-hot-vector

        Returns
        ----------
        predicted_name : str
            the predicted name
        """
        # print(self.target_token_index)
        # print(self.input_token_index)

        # print("input sequence is: " + str(input_seq))
        # the input_seq is a one-hot vector, but length needs be padded with spaces at the end so that sequences are always the sames size
        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_seq)

        # print("states_value is: " + str(states_value))

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, self.target_token_index["\t"]] = 1.0

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ""
        prob_list = []
        while not stop_condition:
            # use dictionary to lookup probabilities and sum up log probabilities     

            # see if output_tokens are probabilties from softmax
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)
            # print("output_tokens: " + str(output_tokens))
            # print("h: " + str(h))
            # print("c: " + str(c))

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            probability = output_tokens[0, -1, sampled_token_index]
            prob_list.append(probability)
            # print(probability)
            # have to safe-guard against the token not being in the dictionary
            # this shouldn't happen when the vocabualry is large enough
            if sampled_token_index in self.reverse_target_char_index:
                sampled_char = self.reverse_target_char_index[sampled_token_index]
            else:
                sampled_char = "\n"
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if sampled_char == "\n" or len(decoded_sentence) > self.max_decoder_seq_length:
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.0

            # Update states
            states_value = [h, c]
            # print("updated states_value is: " + str(states_value))
        return (decoded_sentence, probability)

    """
    trying to have arbitrary batch sizes
    """
    def decode_sequence_batch(self, input_seq_list):
        # print(self.target_token_index)
        # print(self.input_token_index)

        # print("input sequence is: " + str(input_seq))
        # the input_seq is a one-hot vector, but length needs be padded with spaces at the end so that sequences are always the sames size
        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_seq_list)

        # print("states_value is: " + str(states_value))

        # Generate empty target sequence of size number of input rows.
        # Populate the first character of target sequence with the start character.
        target_seq = np.zeros((len(input_seq_list), 1, self.num_decoder_tokens))
        for i in range(len(input_seq_list)):
            target_seq[i, 0, self.target_token_index["\t"]] = 1.0

        # Sampling loop for a batch of sequences
        stop_condition = False
        # this should be a list of strings
        decoded_sentence = ""
        # this should be a list of floats
        prob_list = []
        while not stop_condition:
            # use dictionary to lookup probabilities and sum up log probabilities     

            # see if output_tokens are probabilties from softmax
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)
            # print("output_tokens: " + str(output_tokens))
            # print("h: " + str(h))
            # print("c: " + str(c))

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            probability = output_tokens[0, -1, sampled_token_index]
            prob_list.append(probability)
            # print(probability)
            # have to safe-guard against the token not being in the dictionary
            # this shouldn't happen when the vocabualry is large enough
            if sampled_token_index in self.reverse_target_char_index:
                sampled_char = self.reverse_target_char_index[sampled_token_index]
            else:
                sampled_char = "\n"
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if sampled_char == "\n" or len(decoded_sentence) > self.max_decoder_seq_length:
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.0

            # Update states
            states_value = [h, c]
            # print("updated states_value is: " + str(states_value))
        return (decoded_sentence, probability)

    def predict(self, name:str):
        """
        Given an english name returns the models best inference.
        Uses a trained encoder and decoder model to perform inference.

        Parameters
        ----------
        name : str
            an english name

        Returns
        ----------
        predicted_name : str
            the predicted name
        """
        # need to check here if the length off the name is going to exceed the maximum encoder length
        if len(name) > self.max_encoder_seq_length:
            # return nothing if we cannot handle it
            return ("", 0)
        one_hot_vector = np.zeros(
            (1, self.max_encoder_seq_length, self.num_encoder_tokens), dtype="float32"
        )
        sequence_end = 0
        for t, char in enumerate(name):
            if char in self.input_token_index:
                one_hot_vector[0, t, self.input_token_index[char]] = 1.0
            else:
                # return nothing if we cannot handle it, token has not been encountered before
                return ("", 0)
            sequence_end = t
        # from when the sequence ends, we have to fill up the rest with spaces
        one_hot_vector[0, sequence_end + 1 :, self.input_token_index[" "]] = 1.0
        return self.decode_sequence(one_hot_vector[0:1])

    def create_probabilities(self, data_path):
        """
        given a text file containing name pairs
        this uses the trained model and returns a list of probabilities that relate to how confident the model is on its prediction

        We are not inferring and trying to get the highest probability possible for each name pair!
        We are trying to produce the probability of the screen name given the user name!
        """
        encoder_input, decoder_input, decoder_output = self.processData(data_path)
        print("data is processed")

        # sometimes the data is too big, here we segment it into multiple runs
        prediction_list = []
        # figure out how many runs it will take
        num_runs = int(len(encoder_input) / 1000) + 1
        for i in range(num_runs+1):
            # ummm it crashes when it gets to here
            if i == 21:
                break
            if i != 0:
                slice_range_start = (i-1)*1000
                slice_range_finish = i*1000
                prediction = self.model.predict(
                    [encoder_input[slice_range_start:slice_range_finish], 
                    decoder_input[slice_range_start:slice_range_finish]])
                prediction_list.extend(prediction)
                print("completed prediction iteration: " + str(i) + " of " + str(num_runs))
        print(str(len(prediction_list)) + " predictions completed")

        with open(data_path, "r", encoding="utf-8") as f:
            lines = f.read().split("\n")

        # prediction is a three dimensional numpy array
        # first dimension relates to which row of the original data
        # second dimension is the sequence dimension, length will be the max decoder length
        # third dimension is a probability distribution, length will be the number of target tokens
        prob_list = []
        for i, row in enumerate(prediction_list):
            _, screen_name = lines[i].split('\t')
            screen_name_length = len(screen_name)
            # re-inititialise probability
            probability = 1
            for j, time_step in enumerate(row):
                screen_name_char = screen_name[j]
                if screen_name_char in self.target_token_index:
                    idx = self.target_token_index[screen_name_char]
                else:
                    idx = self.target_token_index[' ']
                probability = probability * time_step[idx]
                # stop when we reach the end of the screen name
                if j == screen_name_length-1:
                    prob_list.append(probability)
                    break
        
        # make sure that for every prediction produced by the model there is a probability associated with it
        assert len(prob_list) == len(prediction_list)
        return prob_list

    def evaluateOnTestData(self, model_name):
        """
        Evaluates trained model on test data.
        Test data is created in the cleansing stage and should be called test1_cleansed.txt, test2_cleansed.txt and test3_cleansed.txt.

        Parameters
        ----------
        model_name : str
            the model that is to be loaded and evaluted upon
        """
        self.model = keras.models.load_model(model_name)

        print("\nevaluating on test set with 0 edit threshold...")
        test_encoder_input, test_decoder_input, test_decoder_output = self.processData('test1_cleansed.txt')
        self.results1 = self.model.evaluate(
            x=[test_encoder_input, test_decoder_input],
            y=test_decoder_output
        )
        print("test loss, test acc:", self.results1)

        print("evaluating on test set with 0.1 edit threshold...")
        test_encoder_input, test_decoder_input, test_decoder_output = self.processData('test2_cleansed.txt')
        self.results2 = self.model.evaluate(
            x=[test_encoder_input, test_decoder_input],
            y=test_decoder_output
        )
        print("test loss, test acc:", self.results2)

        print("evaluating on test set with 0.25 edit threshold...")
        test_encoder_input, test_decoder_input, test_decoder_output = self.processData('test3_cleansed.txt')
        self.results3 = self.model.evaluate(
            x=[test_encoder_input, test_decoder_input],
            y=test_decoder_output
        )
        print("test loss, test acc:", self.results3)


    def determineDimensions(self, path_list:list):
        """
        Very important method so that all text files consumed by this class is understood by the model.
        Given a list of text file names, reads through every single one and saves important parameters the overall model will later use.
        Also saves these important parameters to disk so that a trained model can be reloaded and have all of the neccesary information to perform inference.

        Parameters
        ----------
        path_list : list
            a list of strings, where each string is a path to a text file containing name pairs going to be used by the model
        """
        self.input_characters[" "] = None
        self.target_characters[" "] = None
        for path in path_list:
            # Vectorize the data.
            input_texts = []
            target_texts = []
            
            with open(path, "r", encoding="utf-8") as f:
                lines = f.read().split("\n")
            if self.num_samples is None:
                self.num_samples = 9999999
            for line in lines[: min(self.num_samples, len(lines) - 1)]:
                input_text, target_text = line.split("\t")
                # We use "tab" as the "start sequence" character
                # for the targets, and "\n" as "end sequence" character.
                target_text = "\t" + target_text + "\n"
                input_texts.append(input_text)
                target_texts.append(target_text)
                for char in input_text:
                    if char not in self.input_characters:
                        self.input_characters[char] = None
                for char in target_text:
                    if char not in self.target_characters:
                        self.target_characters[char] = None

            current_num_encoder_tokens = len(self.input_characters)
            current_num_decoder_tokens = len(self.target_characters)
            current_max_encoder_seq_length = max([len(txt) for txt in input_texts])
            current_max_decoder_seq_length = max([len(txt) for txt in target_texts])

            if self.num_encoder_tokens < current_num_encoder_tokens:
                self.num_encoder_tokens = current_num_encoder_tokens
            if self.num_decoder_tokens < current_num_decoder_tokens:
                self.num_decoder_tokens = current_num_decoder_tokens
            if self.max_encoder_seq_length < current_max_encoder_seq_length:
                self.max_encoder_seq_length = current_max_encoder_seq_length
            if self.max_decoder_seq_length < current_max_decoder_seq_length:
                self.max_decoder_seq_length = current_max_decoder_seq_length
        print("Number of unique input tokens:", self.num_encoder_tokens)
        print("Number of unique output tokens:", self.num_decoder_tokens)
        print("Max sequence length for inputs:", self.max_encoder_seq_length)
        print("Max sequence length for outputs:", self.max_decoder_seq_length)

        # save the data paramaters so that it can be loaded into another instance of this class
        with open('data_parameters', 'w') as f:
            f.write("num_unique_input_tokens:\t" + str(self.num_encoder_tokens) + '\n')
            f.write("num_unique_output_tokens:\t" + str(self.num_decoder_tokens) + '\n')
            f.write("max_seq_length_of_input:\t" + str(self.max_encoder_seq_length) + '\n')
            f.write("max_seq_length_of_output:\t" + str(self.max_decoder_seq_length) + '\n')
        
        # save the input and output character set to disk
        with open('input_char_set', 'wb') as f:
            pickle.dump(self.input_characters, f)
        with open('output_char_set', 'wb') as f:
            pickle.dump(self.target_characters, f)

    def loadDataParameters(self):
        """
        Load in previously determined model parameters to current instance of class.
        Requires to have already called determineDimensions() method in a previous run.
        """
        with open ('input_char_set', 'rb') as f:
            self.input_characters = pickle.load(f)
        with open ('output_char_set', 'rb') as f:
            self.target_characters = pickle.load(f)
        with open('data_parameters', 'r') as f:
            lines = f.read().split('\n')
        self.num_encoder_tokens = int(lines[0].split('\t')[1])
        self.num_decoder_tokens = int(lines[1].split('\t')[1])
        self.max_encoder_seq_length = int(lines[2].split('\t')[1])
        self.max_decoder_seq_length = int(lines[3].split('\t')[1])

        # re-create these dictionaries after data is loaded in
        self.input_token_index = dict([(char, i) for i, char in enumerate(self.input_characters.keys())])
        # self.target_token_index = dict([(char, i) for i, char in enumerate(self.target_characters)])
        self.target_token_index = {}
        for i, char in enumerate(self.target_characters.keys()):
            self.target_token_index[char] = i


    def runWholeTrainProcess(self, train_data_path, model_name):
        """
        Chains together all of the methods that trains a model.
        Just a convenient way to call all the functions in one go.

        Parameters
        ----------
        train_data_path : str
            The data path to where the training text file is located. This is required as the model needs to know which file to train on.
        model_name : str
            The name that the trained model file will be called and saved as.
        """
        # need to first determine the proper dimensions of the model we are going to be using
        self.determineDimensions([train_data_path, 'test1_cleansed.txt', 'test2_cleansed.txt', 'test3_cleansed.txt'])

        train_encode_input, train_decode_input, train_decode_output = self.processData(train_data_path)

        self.buildModel()

        self.trainModel(model_name, train_encode_input, train_decode_input, train_decode_output)

        self.createDecoderEncoder(model_name)





    """ ------------------- PLOTTING AND STATS FUNCTIONS HERE --------------------------"""
    def plotAccuracy(self, file_name = None):
        # summarize history for accuracy
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        if file_name is None:
            plt.savefig(self.language+"_accuracy_" + str(self.epochs)+"_epochs.png")
        else:
            plt.savefig(file_name)
        plt.show()
    
    def plotLoss(self, file_name = None):
        # summarize history for loss
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        if file_name is None:
            plt.savefig(self.language+"_loss_"+str(self.epochs)+"_epochs.png")
        else:
            plt.savefig(file_name)
        plt.show()

    def saveTrainingStats(self, file_name=None):
        min_loss = min(self.history.history['val_loss'])
        max_accuracy = max(self.history.history['val_accuracy'])
        min_loss_epoch = self.history.history['val_loss'].index(min(self.history.history['val_loss']))
        max_accuracy_epoch = self.history.history['val_accuracy'].index(max(self.history.history['val_accuracy']))
        time_taken = self.elapsed_time
        if file_name is None:
            file_name = self.language+"_stats_"+str(self.epochs)+"_epochs.txt"
        with open(file_name, 'w') as f:
            f.write("number of epochs: " + str(self.epochs) + '\n')
            f.write("time taken: " + str(datetime.timedelta(seconds=self.elapsed_time)) + '\n')
            f.write("min validation loss: " + str(min_loss) + '\n')
            f.write("min validation loss epoch: " + str(min_loss_epoch) + '\n')
            f.write("max validation accuracy: " + str(max_accuracy) + '\n')
            f.write("max validation accuracy epoch: " + str(max_accuracy_epoch) + '\n')

            f.write("test set 1 loss: " + str(self.results1[0]) + '\n')
            f.write("test set 1 accuracy: " + str(self.results1[1]) + '\n')
            f.write("test set 2 loss: " + str(self.results2[0]) + '\n')
            f.write("test set 2 accuracy: " + str(self.results2[1]) + '\n')
            f.write("test set 3 loss: " + str(self.results3[0]) + '\n')
            f.write("test set 3 accuracy: " + str(self.results3[1]) + '\n')