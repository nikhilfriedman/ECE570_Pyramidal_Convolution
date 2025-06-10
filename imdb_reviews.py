import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from imdb_dataloader import save_glove_binary, preprocess_and_save_dataset, load_glove_binary, load_preprocessed_dataset, IMDbPreprocessedDataset, collate_batch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

class PyConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PyConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=1, padding=0)
        self.conv3 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        x = torch.cat([x1, x3, x5, x7], dim=1)
        return F.relu(self.bn(x))

class EnhancedCNNLSTMWithPyConv(nn.Module):
    def __init__(self, embed_dim, hidden_dim=256, num_classes=2, lstm_layers=2, dropout=0.5):
        super(EnhancedCNNLSTMWithPyConv, self).__init__()
        self.pyconv1 = PyConvBlock(embed_dim, 512)
        self.pyconv2 = PyConvBlock(512, 256)
        self.pyconv3 = PyConvBlock(256, 128)
        self.pyconv4 = PyConvBlock(128, 64)
        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(64, hidden_dim, num_layers=lstm_layers, bidirectional=True, batch_first=True)

        self.fc1 = nn.Linear(hidden_dim * 2, 128)  # Increased size
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.dropout(self.pyconv1(x))
        x = F.max_pool1d(x, kernel_size=2)
        x = self.dropout(self.pyconv2(x))
        x = F.max_pool1d(x, kernel_size=2)
        x = self.dropout(self.pyconv3(x))
        x = F.max_pool1d(x, kernel_size=2)
        x = self.dropout(self.pyconv4(x))
        x = F.adaptive_max_pool1d(x, 1).squeeze(2)

        x = x.unsqueeze(1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct, total = 0, 0
    
    for text, label in train_loader:
        text, label = text.to(device).float(), label.to(device)
        
        optimizer.zero_grad()
        output = model(text)
        loss = criterion(output, label)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # added

        optimizer.step()
        
        total_loss += loss.item()
        preds = output.argmax(dim=1)
        correct += (preds == label).sum().item()
        total += label.size(0)
    
    train_loss = total_loss / len(train_loader)
    train_accuracy = correct / total
    return train_loss, train_accuracy

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct, total = 0, 0
    
    with torch.no_grad():
        for text, label in test_loader:
            text, label = text.to(device).float(), label.to(device)
            output = model(text)
            loss = criterion(output, label)
            total_loss += loss.item()
            
            preds = output.argmax(dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)
    
    test_loss = total_loss / len(test_loader)
    test_accuracy = correct / total
    return test_loss, test_accuracy

if __name__ == "__main__":
    embed_dim = 100
    batch_size = 64
    num_epochs = 50
    learning_rate = 0.00005
    glove_path = 'glove.6B.100d.txt'
    binary_path = 'glove_embeddings.npz'
    preprocessed_train_path = 'preprocessed_train.pkl'
    preprocessed_test_path = 'preprocessed_test.pkl'
    train_dir = 'aclImdb/train'
    test_dir = 'aclImdb/test'
    weight_decay = 5e-5
    eta_min = 1e-5
    dropout = 0.7

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    create_binaries = True # used to speedup loading for training process
    
    if(create_binaries):
        print('Creating glove embeddings...')
        save_glove_binary(glove_path, binary_path)
        glove_embeddings = load_glove_binary(binary_path)
        print('Preprocessing data...')
        preprocess_and_save_dataset(train_dir, glove_embeddings, embed_dim, preprocessed_train_path)
        preprocess_and_save_dataset(test_dir, glove_embeddings, embed_dim, preprocessed_test_path)

    print('Loading data...')

    train_data = load_preprocessed_dataset(preprocessed_train_path)
    test_data = load_preprocessed_dataset(preprocessed_test_path)
    train_dataset = IMDbPreprocessedDataset(train_data)
    test_dataset = IMDbPreprocessedDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    print('Creating model...')

    model = EnhancedCNNLSTMWithPyConv(embed_dim=embed_dim, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # changeed frorm 1e-3
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    print('Using device:', device)

    last_test_acc = 0
    best_acc = 0
    patience = 5
    no_imrovement = 0

    print(f'batch_size = {batch_size}, learning_rate = {learning_rate}, weight_decay = {weight_decay}, eta_min = {eta_min}, dropout = {dropout}, scheduler = {scheduler.__class__}')

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        scheduler.step()

        if(test_acc > best_acc):
            best_acc = test_acc
            no_imrovement = 0
        else:
            no_imrovement += 1

        if(no_imrovement == patience):
            print(f'Early stop at epoch {epoch + 1}')
            break

        acc_diff = test_acc - last_test_acc

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train Accuracy = {train_acc:.4f}, Test Loss = {test_loss:.4f}, Test Accuracy = {test_acc:.4f}, Test Accuracy Diff = {acc_diff:.4f}")

        last_test_acc = test_acc

    torch.save(model.state_dict(), 'trained.pth')
