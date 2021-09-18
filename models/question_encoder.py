import torch
import torch.nn as nn
import torch.nn.functional as F


class QuestionEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, out_size):
        super(QuestionEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.out_size = out_size

        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(self.embed_size, self.out_size, batch_first=True)

    def forward(self, x):
        """Implements the class to encode the questions into feature vectors. Uses embeddings and LSTMs for the same
        
        Args:
            x (N, seq_len, embed_size): Tensor containing numbers corresponding to words of the sentence.
            self.embed(x): (N, seq_len, embed_size)
        
        Returns:
            c (N, out_size): Feature vector corresponding to each question vector
        """
        # output, (h, c) = self.lstm(self.embed(x))
        output, (h, c) = self.lstm(x)
        return torch.squeeze(h, dim=0)


if __name__ == '__main__':
    vocab_size, embed_size, out_size = 500, 250, 90
    model = QuestionEncoder(vocab_size, embed_size, out_size)
    # batch of 2 sentences, each containing 4 words
    sentence = torch.tensor([
                             [1, 2, 3, 4],
                             [5, 6, 7, 8]
    ])
    out = model(sentence)
    assert tuple(out.shape) == (2, out_size)
    
    
    