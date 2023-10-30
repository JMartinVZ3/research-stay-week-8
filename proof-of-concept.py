import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Training pairs
training_pairs = [
    ("Can I help you", "¿Puedo ayudarte"),
    ("Where would you like to travel", "¿Adónde te gustaría viajar"),
    ("When do you want to travel", "¿Cuándo quieres viajar"),
    ("How many people are travelling", "¿Cuántas personas viajan"),
    ("I have booked a ticket for you", "He reservado un billete para ti"),
]

# Build vocabularies
input_vocab = set()
output_vocab = set()
for pair in training_pairs:
    input_vocab.update(pair[0].split())
    output_vocab.update(pair[1].split())

input_vocab_size = len(input_vocab)
output_vocab_size = len(output_vocab)

input_word2idx = {word: idx for idx, word in enumerate(input_vocab)}
output_word2idx = {word: idx for idx, word in enumerate(output_vocab)}


# Convert sentences to tensors
def sentence_to_tensor(sentence, vocab):
    return torch.tensor([vocab[word] for word in sentence.split()])


# Encoder model
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        return hidden


# Decoder model
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        x = x.unsqueeze(0)
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        output = self.out(output.squeeze(0))
        return output, hidden


# Encoder-Decoder model
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        target_len = trg.shape[0]
        target_vocab_size = self.decoder.out.out_features

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(trg.device)
        encoder_hidden = self.encoder(src)

        decoder_input = trg[0, :]
        decoder_hidden = encoder_hidden

        use_teacher_forcing = (
            True if torch.rand(1).item() < teacher_forcing_ratio else False
        )

        for t in range(1, target_len):
            output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = output
            if use_teacher_forcing:
                decoder_input = trg[t]
            else:
                decoder_input = output.argmax(1)
        return outputs


training_data = [
    (
        sentence_to_tensor(pair[0], input_word2idx),
        sentence_to_tensor(pair[1], output_word2idx),
    )
    for pair in training_pairs
]

# Continue as in the previous code to define the Encoder, Decoder, and EncoderDecoder classes

# Training
encoder_hidden_size = 64
decoder_hidden_size = 64

encoder = Encoder(input_vocab_size, encoder_hidden_size)
decoder = Decoder(decoder_hidden_size, output_vocab_size)

model = EncoderDecoder(encoder, decoder)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100

for epoch in range(num_epochs):
    for src, trg in training_data:
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")


# Translating new sentences
def translate_sentence(sentence, model, input_vocab, output_vocab):
    with torch.no_grad():
        model.eval()
        src_tensor = sentence_to_tensor(sentence, input_vocab)
        src_tensor = src_tensor.unsqueeze(1).to(next(model.parameters()).device)
        encoder_hidden = model.encoder(src_tensor)
        trg_indexes = [output_word2idx["<SOS>"]]
        for _ in range(10):
            trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(
                next(model.parameters()).device
            )
            with torch.no_grad():
                output, encoder_hidden = model.decoder(trg_tensor, encoder_hidden)
            pred_token = output.argmax(1).item()
            trg_indexes.append(pred_token)
            if pred_token == output_word2idx["<EOS>"]:
                break
        trg_tokens = [output_vocab[idx] for idx in trg_indexes]
        return trg_tokens[1:]


# Test translation
test_sentence = "Can I help you"
translated_sentence = translate_sentence(
    test_sentence, model, input_word2idx, output_word2idx
)
print(f"English: {test_sentence}")
print(f"Spanish: {' '.join(translated_sentence)}")

"""
Honestly I couldnt make it work since there were errors with the original classes
But the idea behind this proof of concept in a business scenario was that
we might have a travel booking service that with this code can now automate some of their customer interaction 
in Spanish with the use of this translation model. They feed in English sentences and get 
the relevant Spanish translations in return. This can increase the efficiency of the company.
"""
