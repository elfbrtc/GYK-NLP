import torch    
import torch.nn as nn
import torch.nn.functional as F

sectences = [
    "ali okula gitti",
    "veli sinemaya gitti",
    "ayşe markete gitti",
    "fatma işe gitti"

]


#1. Tokenization (manuel)

vocab = { "< PAD >" : 0}

index = 1

for sentence in sectences:
    for word in sentence.split():
        if word not in vocab:
            vocab[word] = index
            index += 1


#encode()

def encode(sentence):
    tokens = sentence.split()
    return torch.tensor([vocab[token] for token in tokens[:-1]], dtype = torch.long), torch.tensor(vocab[tokens[-1]], dtype = torch.long)


data = [encode(sentence) for sentence in sectences]

print(data[0])

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, d_model)

        self.pos_embed = nn.Parameter(torch.zeros(1, 3, d_model))

        self.attn = nn.MultiheadAttention(embed_dim = d_model, num_heads = 1, batch_first = True)
        # Head --- Attentionda kaç farklı dikkat edicelecek durum? (odak noktası)
        #özne-fiil
        #sıfat-isim
        # özne özne bağı

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, vocab_size)
        )

    def forward(self, x):
        x = self.embed(x) + self.pos_embed[:, :x.size(1), :]
        
        #Q, K, V

        #Q --- Ne arıyorum?
        #K --- Hangi bilgilerin keylerini karşılaştıracağım?
        #V --- Sonuç olarak hangi bilgileri çekeceğim?

        attn_output, _ = self.attn(x, x, x)

        out = self.ff(attn_output[:, -1, :])

        return out
    
model = MiniGPT(len(vocab), d_model = 32)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(200):
    total_loss = 0
    for x, y in data:
        x = x.unsqueeze(0)

        out = model(x)

        loss = loss_fn(out, y.unsqueeze(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

#predict fonksiyonu yazılacak

def predict(sentence):
    # Cümleyi tokenize et
    tokens = sentence.split()
    input_tokens = torch.tensor([vocab[token] for token in tokens], dtype=torch.long)
    input_tokens = input_tokens.unsqueeze(0)  # batch dimension ekle
    
    # Model ile tahmin yap
    model.eval()  # evaluation mode
    with torch.no_grad():
        output = model(input_tokens)
        predicted_index = torch.argmax(output, dim=-1).item()
    
    # Index'i kelimeye çevir
    for word, idx in vocab.items():
        if idx == predicted_index:
            return word
    return "Unknown"

# Test et
test_sentences = [
    "ali okula",
    "veli sinemaya", 
    "ayşe markete",
    "fatma işe"
]

print("\nTahminler:")
for sentence in test_sentences:
    next_word = predict(sentence)
    print(f"'{sentence}' -> '{next_word}'")



        



