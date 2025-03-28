import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import argparse

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import *
from utilities import Utilities

seed = 42

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts



def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(encoder, classifier, data_loader):
    """Compute the accuracy of the classifier on the data in data_loader."""
    encoder.eval()
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            mask = (X != 0).float().to(device)

            # Forward pass through encoder
            encoder_output, _ = encoder(X)
            x = (encoder_output * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)

            # Forward pass through classifier
            outputs = classifier(x)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
    accuracy = (100 * total_correct / total_samples)
    encoder.train()
    classifier.train()
    return accuracy

def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    total_loss = 0.0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for i, (X, Y) in enumerate(data_loader):
            X, Y = X.to(device), Y.to(device)
            logits, _ = decoderLMmodel(X)
            logits = logits.view(-1, logits.size(-1))
            Y = Y.view(-1)
            loss = criterion(logits, Y)
            total_loss += loss.item()
            total_tokens += Y.size(0)
            if (i + 1) >= eval_iters:
                break
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    decoderLMmodel.train()
    return perplexity.item()


def train_CLS_model(tokenizer, train_CLS_loader, test_CLS_loader, epochs_CLS, method='standard'):
    """
    Train the classifier model on the training data and evaluate on the test data.
    """
    def create_mask(x):
        # x: (batch_size, seq_len)
        mask = (x != 0).float()
        return mask

    # Initialize the Encoder
    encoder = TransformerEncoder(
        vocab_size=tokenizer.vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        max_seq_len=block_size,
        method=method
    ).to(device)

    # Initialize the Classifier
    classifier = FNNClassifier(
        n_input=n_embd,
        n_hidden=n_hidden,
        n_output=n_output
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=learning_rate,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    for epoch in range(epochs_CLS):
        epoch_loss = 0
        total_correct = 0
        total_samples = 0

        for xb, yb in train_CLS_loader:
            xb, yb = xb.to(device), yb.to(device)
            mask = create_mask(xb)

            encoder_output, _ = encoder(xb)  # Encode the input
            x = (encoder_output * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)   # Average the encoder output
            outputs = classifier(x)          # Get the classifier output

            loss = criterion(outputs, yb)    # Loss

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute statistics
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == yb).sum().item()
            total_samples += yb.size(0)

        epoch_accuracy = 100 * total_correct / total_samples
        epoch_loss_avg = epoch_loss / len(train_CLS_loader)

        # Compute test accuracy
        test_accuracy = compute_classifier_accuracy(encoder, classifier, test_CLS_loader)
        print(f"Epoch [{epoch+1}/{epochs_CLS}], Loss: {epoch_loss_avg:.4f}, "
              f"Train Accuracy: {epoch_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")

        # Adjust learning rate
        scheduler.step(epoch_loss_avg)
    
    return encoder, classifier

def train_LM_model(tokenizer, train_LM_loader, test_LM_loaders, max_iters, method='standard'):
    """
    Train the decoder model on the training data and evaluate on the test data.
    """
    # Initialize the Decoder
    decoder = TransformerDecoder(
        vocab_size=tokenizer.vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        max_seq_len=block_size,
        method=method
    ).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        decoder.parameters(),
        lr=learning_rate
    )

    # Training loop
    total_steps = 0
    total_loss = 0.0
    decoder.train()

    for i, (xb, yb) in enumerate(train_LM_loader):
        if total_steps >= max_iters:
            break
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()

        # Forward pass through the decoder
        logits, _ = decoder(xb)
        logits = logits.view(-1, logits.size(-1))
        yb = yb.view(-1)

        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_steps += 1

        # Every 100 iterations, compute and print perplexity
        if total_steps % 100 == 0:
            avg_loss = total_loss / total_steps
            train_perplexity = compute_perplexity(decoder, train_LM_loader, eval_iters)
            print(f"Iteration [{total_steps}/{max_iters}], "
                  f"Loss: {avg_loss:.4f}, Train Perplexity: {train_perplexity:.2f}")

            # Compute test perplexities
            for name, test_loader in test_LM_loaders.items():
                test_perplexity = compute_perplexity(decoder, test_loader, eval_iters)
                print(f"Test Perplexity on {name}: {test_perplexity:.2f}")
            print("-" * 50)

    # Return the trained decoder
    return decoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-m", type=str, default="part1", help="Choose the part of the assignment to run, 'part1' or 'part2' or 'part3'")
    args = parser.parse_args()

    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

  
    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

    # for the classification task, you will train for a fixed number of epochs like this:
    # CLS training code here
    if args.mode == "part1":
        test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
        test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=False)
        encoder, classifier = train_CLS_model(tokenizer, train_CLS_loader, test_CLS_loader, epochs_CLS)
        # 1.4 Sanity Check
        utilities = Utilities(tokenizer, encoder)
        sentence = "This task is interesting but hard, making me feel tired."
        utilities.sanity_check(sentence, block_size)

    # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
    # LM training code here
    elif args.mode == "part2":
        test_files = {
            "Obama": "speechesdataset/test_LM_obama.txt",
            "H_Bush": "speechesdataset/test_LM_hbush.txt",
            "W_Bush": "speechesdataset/test_LM_wbush.txt"
        }
        test_LM_loaders = {}
        for name, filepath in test_files.items():
            with open(filepath, 'r', encoding='utf-8') as f:
                test_text = f.read()
            test_dataset = LanguageModelingDataset(tokenizer, test_text, block_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            test_LM_loaders[name] = test_loader
        decoder = train_LM_model(tokenizer, train_LM_loader, test_LM_loaders, max_iters)
        utilities = Utilities(tokenizer, decoder)
        sentence = "Let us see what will be the next word predicted."
        utilities.sanity_check(' '.join(sentence.split()[:-1]), block_size)

    # Test the special methods
    elif args.mode == "part3":
        test_files = {
            "Obama": "speechesdataset/test_LM_obama.txt",
            "H_Bush": "speechesdataset/test_LM_hbush.txt",
            "W_Bush": "speechesdataset/test_LM_wbush.txt"
        }
        test_LM_loaders = {}
        for name, filepath in test_files.items():
            with open(filepath, 'r', encoding='utf-8') as f:
                test_text = f.read()
            test_dataset = LanguageModelingDataset(tokenizer, test_text, block_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            test_LM_loaders[name] = test_loader
        print("-" * 80)
        print("The special methods are train on the decoding task.")
        print("-" * 80)
        # Positional Encoding: Alibi method
        print("-" * 30, "Alibi Method", "-" * 30)
        decoder = train_LM_model(tokenizer, train_LM_loader, test_LM_loaders, max_iters, method='alibi')
       
        # Sparse Attention Patterns: Local Window Method, Blockwise method
        print("-" * 30, "Local Window Method", "-" * 30)
        decoder = train_LM_model(tokenizer, train_LM_loader, test_LM_loaders, max_iters, method='local')
        print("-" * 30, "Blockwise Method", "-" * 30)
        decoder = train_LM_model(tokenizer, train_LM_loader, test_LM_loaders, max_iters, method='block')
    
        # Disentangled Attention Patterns method
        print("-" * 30, "Disentangled Attention Patterns Method", "-" * 30)
        decoder = train_LM_model(tokenizer, train_LM_loader, test_LM_loaders, max_iters, method='disentangled')
        
        # Other special methods
        print("-" * 30, "Performer Attention Method", "-" * 30)
        decoder = train_LM_model(tokenizer, train_LM_loader, test_LM_loaders, max_iters, method='performer')
        
        test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
        test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=False)
        print("-" * 80)
        print("The special methods are train on the classification task.")
        print("-" * 80)
        # Positional Encoding: Alibi method
        print("-" * 30, "Alibi Method", "-" * 30)
        encoder, classifier = train_CLS_model(tokenizer, train_CLS_loader, test_CLS_loader, epochs_CLS, method='alibi')

        # Sparse Attention Patterns: Local Window Method, Blockwise method
        print("-" * 30, "Local Window Method", "-" * 30)
        encoder, classifier = train_CLS_model(tokenizer, train_CLS_loader, test_CLS_loader, epochs_CLS, method='local')
        print("-" * 30, "Blockwise Method", "-" * 30)
        encoder, classifier = train_CLS_model(tokenizer, train_CLS_loader, test_CLS_loader, epochs_CLS, method='block')

        # Disentangled Attention Patterns method
        print("-" * 30, "Disentangled Attention Patterns Method", "-" * 30)
        encoder, classifier = train_CLS_model(tokenizer, train_CLS_loader, test_CLS_loader, epochs_CLS, method='disentangled')

        # Other special methods
        print("-" * 30, "Performer Attention Method", "-" * 30)
        encoder, classifier = train_CLS_model(tokenizer, train_CLS_loader, test_CLS_loader, epochs_CLS, method='performer')

    # Invalid mode
    else:
        print("Invalid mode. Please choose 'part1', 'part2' or 'part3' by using the --mode or -m argument.")

if __name__ == "__main__":
    main()
