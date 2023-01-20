
#importing libraries
import nltk
import re
import string
from math import log
from tabulate import tabulate
import sys

nltk.download('punkt')

smoothing = int(sys.argv[2])
filename = sys.argv[3]

#loading data
def load_data(filename):
  with open(filename, "r") as f:
    data = f.read()
  return data

training_data = load_data(filename)

def preprocess_data(data):
  data.lower()
  return data.translate(str.maketrans('','',string.punctuation))

training_data = preprocess_data(training_data)
print(training_data)

# Segmenting into sentences
def segment_into_sentences(text):
  sentences = text.split('\n')
  sentences = [sentence.strip() for sentence in sentences]
  sentences = [sentence for sentence in sentences if len(sentence)>0]
  return sentences

training_data = segment_into_sentences(training_data)
print(training_data)

#Tokenizing sentences
def tokenize_sentences(sentences):
  tokenized_sentences = []
  for sentence in sentences:
    tokenized_sentence = nltk.word_tokenize(sentence)
    tokenized_sentences.append(tokenized_sentence)
  return tokenized_sentences

training_data = tokenize_sentences(training_data)
print(training_data)

#Vocabulary creation
def create_vocabulary(data):
  vocabulary = {}
  for sentence in data:
    for word in sentence:
      if word in vocabulary.keys():
        vocabulary[word]+=1
      else:
        vocabulary[word]=1
  return vocabulary

word_counts = create_vocabulary(training_data)
print(word_counts)
print("Vocabulary: ", word_counts.keys())
print("Number of unique words: ", len(word_counts))

#Handling unknown words
count_threshold = 2
#Gets the Vocabulary of words whose frequency is greater than the count_threshold
def get_closed_vocabulary(vocabulary, count_threshold):
  closed_vocabulary = []
  for word in vocabulary:
    if vocabulary[word] >= count_threshold:
      closed_vocabulary.append(word)
  return closed_vocabulary

closed_vocabulary = get_closed_vocabulary(word_counts, count_threshold)
print(closed_vocabulary)
print(len(closed_vocabulary))

#Replacing unknown words in the data with '<unk>' token
unknown_token = '<unk>'
def replace_unknown_words(data, closed_vocabulary, unknown_token):
  replaced_data = []
  for sentence in data:
    replaced_sentence = []
    for word in sentence:
      if word in closed_vocabulary:
        replaced_sentence.append(word)
      else:
        replaced_sentence.append(unknown_token)
    replaced_data.append(replaced_sentence)
  return replaced_data

training_data = replace_unknown_words(training_data, closed_vocabulary, unknown_token)
print(training_data)
word_counts = create_vocabulary(training_data)
print(word_counts)
print("Vocabulary: ", word_counts.keys())
print("Number of unique words: ", len(word_counts))

#Appending start and end tokens
def append_start_and_end_tokens(data):
  start_token = '<s>'
  end_token = '</s>'
  data_with_tokens = []
  for sentence in data:
    sentence_with_tokens = [start_token] + sentence + [end_token]
    data_with_tokens.append(tuple(sentence_with_tokens))
  return data_with_tokens

training_data = append_start_and_end_tokens(training_data)
print(training_data)
unigrams = create_vocabulary(training_data)
print(unigrams)

#Generating Bigrams
def generate_bigrams(data):
  bigrams = {}
  for sentence in data:
    for i in range(0, len(sentence)):
      bigram = sentence[i:i+2]
      if bigram in bigrams.keys():
        bigrams[bigram]+=1
      else:
        bigrams[bigram]=1
  return bigrams

bigrams = generate_bigrams(training_data)
print(bigrams)

#Bigram probabilities
def compute_bigram_probabilities(bigram_frequencies, unigram_frequencies):
  bigram_probabilities = {}
  for bigram in bigram_frequencies:
    bigram_probabilities[bigram] = bigram_frequencies[bigram]/unigram_frequencies[bigram[0]]
  return bigram_probabilities

def compute_bigram_probabilities_smoothing(bigram_frequencies, unigram_frequencies):
  bigram_probabilities = {}
  for bigram in bigram_frequencies:
    bigram_probabilities[bigram] = (bigram_frequencies[bigram] + 1)/(unigram_frequencies[bigram[0]]+1)
  return bigram_probabilities

bigram_probabilities = compute_bigram_probabilities(bigrams, unigrams)
print(bigram_probabilities)

#Smoothing
def smoothing(smoothing, bigram_frequencies, unigram_frequencies, bigram):
  if smoothing == 0:
    return bigram_frequencies[bigram]/unigram_frequencies[bigram[0]]
  elif smoothing == 1:
    return (bigram_frequencies[bigram]+1)/(unigram_frequencies[bigram[0]]+1)

#Preparing the test data
test_data = "mark antony , heere take you caesars body : you shall not come to them poet ." + "\n" + "no , sir , there are no comets seen , the heauens speede thee in thine enterprize ."
test_data = preprocess_data(test_data)
test_data = segment_into_sentences(test_data)
test_data = tokenize_sentences(test_data)
test_data = append_start_and_end_tokens(test_data)
test_unigrams = create_vocabulary(test_data)
test_bigrams = generate_bigrams(test_data)
test_bigram_probabilities = compute_bigram_probabilities(test_bigrams, test_unigrams)
test_bigram_probabilities_smoothing = compute_bigram_probabilities_smoothing(test_bigrams, test_unigrams)
print("Test data\n", test_data)
print("Unigrams\n", test_unigrams)
print("Bigrams\n", test_bigrams)
print("Bigram Probabilities when smoothing = 0", test_bigram_probabilities)
print("Bigram Probabilities when smoothing = 1", test_bigram_probabilities_smoothing)

#Final Solution
print("A table showing the bigram counts for both sentences")
print(tabulate([test_bigrams.keys(), test_bigrams.values()], tablefmt="fancy_grid" ))

print("A table showing the bigram probabilities for both sentences")
def create_bigram_probability_table(bigrams, unigrams, probabilities):
    nrow = len(bigrams)
    ncol = len(unigrams)
    table = []    
    col1 = list(bigrams.keys())
    row1 = list(unigrams.keys())
    table.append([0] + row1)
    for i in range(0, nrow):      
      table.append([col1[i]] + [0]*ncol)
    for bigram, probability in probabilities.items():
        row_index = col1.index(bigram) + 1
        column_index = row1.index(bigram[0]) + 1
        table[row_index][column_index] = probability
    return table
print("smoothing = 0")
print(tabulate(create_bigram_probability_table(test_bigrams, test_unigrams, test_bigram_probabilities),  tablefmt="fancy_grid"))
print("smoothing = 1")
print(tabulate(create_bigram_probability_table(test_bigrams, test_unigrams, test_bigram_probabilities_smoothing),  tablefmt="fancy_grid"))

print("The probability of both sentences under the trained model")
for sentence in test_data:
  probability_smoothing_0 = 1
  probability_smoothing_1 = 1
  for i in range(0, len(sentence)):
    bigram = sentence[i:i+2]
    probability_smoothing_0 += log(test_bigram_probabilities[bigram], 10)
    probability_smoothing_1 += log(test_bigram_probabilities_smoothing[bigram], 10)
  print("Test sentence:",sentence)
  print("Probabilty without smoothing:", probability_smoothing_0)
  print("Probabilty with smoothing:", probability_smoothing_1, "\n" )