class Vocab():
  """
  Functions:
  give it a list of sentences, phase, single word (token)
  
  it will create a vocab using the tokens
  """
  def __init__(self):
        self.PAD_ = 0   # Used for padding short sentences
        self.SOS_ = 1   # Start-of-sentence token
        self.EOS_ = 2   # End-of-sentence token

        self.word2index = {}
        self.word2count = {}
        self.index2word = {self.PAD_: "PAD", self.SOS_: "SOS", self.EOS_: "EOS"}
        self.num_words = 3
        self.num_sentences = 0
        self.longest_sentence = 0 # all the sentences with length less then this will be padded using the token PAD

  def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else: # word2count stores the total tokens in the corpus
            self.word2count[word] += 1
            
  def add_sentence(self, sentence):
        sentence_len = 0
        for word in sentence.split(' '):
            sentence_len += 1
            self.add_word(word)
        if sentence_len > self.longest_sentence:
            self.longest_sentence = sentence_len
        self.num_sentences += 1

  def to_word(self, index): # lookup table 
        return self.index2word[index]

  def to_index(self, word): # lookup table 
        return self.word2index[word]
