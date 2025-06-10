import torch
from imdb_reviews import EnhancedCNNLSTMWithPyConv
import numpy as np
from imdb_dataloader import load_glove_binary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_dim = 100
model = EnhancedCNNLSTMWithPyConv(embed_dim=embed_dim).to(device)
model.load_state_dict(torch.load('trained.pth', map_location=device, weights_only=True))
model.eval()

glove_path = 'glove_embeddings.npz'
glove_embeddings = load_glove_binary(glove_path)

def preprocess_text(text, glove_embeddings, embed_dim=100):
    tokens = text.lower().split()
    embeddings = []
    for token in tokens:
        if token in glove_embeddings:
            embeddings.append(glove_embeddings[token])
        else:
            embeddings.append(np.zeros(embed_dim))
    embeddings = embeddings[:50] + [np.zeros(embed_dim)] * (50 - len(embeddings))

    embeddings_array = np.array(embeddings)
    return torch.tensor(embeddings_array).unsqueeze(0)

user_input = input("Enter a movie review: ")

input_tensor = preprocess_text(user_input, glove_embeddings).to(device).float()

with torch.no_grad():
    output = model(input_tensor)
    predicted_class = output.argmax(dim=1).item()

if(predicted_class):
    print('Predicted Class: Positive')
else:
    print('Predicted Class: Negative')