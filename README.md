# CSE256 Fall 2024 Assignment 2 by Haozhou Xu
This project implements features related to Transformer, including:

1. Implementation of Attention and Encoding layers.
2. Implementation of Decoding layers.
3. Implementation of Special requirements, including Positional Encoding (Alibi attention), Sparse Attention Patterns (Local window attention and Blockwise attention), Disentangled Attention Patterns (Disentangled attention), and Other Architectural Changes (Performer attention).

## Author
Haozhou Xu

## Requirements
Python 3.x
PyTorch
matplotlib
nltk
To install the necessary dependencies, run:

```bash
pip install matplotlib torch nltk
```

Please also make sure to put the following files in the /speechesdataset directory:
if no such directory exists, create one and put the following files in it.
```bash
mkdir speechesdataset
mv train_CLS.tsv speechesdataset/
mv test_CLS.tsv speechesdataset/
mv train_LM.txt speechesdataset/
mv test_LM_hbush.txt speechesdataset/
mv test_LM_obama.txt speechesdataset/
mv test_LM_wbush.txt speechesdataset/
```

## Usage
You can run tasks corresponding to above mentioned three tasks by using the argument below:

Command-line Arguments:
* --mode (-m): Specify the task to run. Options include part1, part2, and part3.

## Running the Tasks:
* Task 1: Implementing Encoder, train and test the model on CLS task with visualizations of attention weights.
```bash
python main.py --mode part1
# or
python main.py -m part1
```

* Task 2: Implementing Decoder, train and test the model on LM task with visualizations of attention weights.
```bash
python main.py --mode part2
# or
python main.py -m part2
```

* Task 3: Implementing Special Requirements, train and test the model on both LM and CLS tasks, no visualizations.
```bash
python main.py --mode part3
# or
python main.py -m part3
```

## File Structure
- main.py: Contains the main function to run experiments and evaluate models.
- dataset.py: Contains the Dataset class for loading and processing the data.
- tokenizer.py: Contains the Tokenizer class for tokenizing the data.
- utilities.py: Contains utility functions for attention visualization and plotting.
- transformer.py: Contains the Transformer model classes.
- speechdataset: Contains the dataset files for training and testing.

Output

During training, the models will output the following:

- Training and development accuracy/perplexity printed after every epoch/hundred iterations.

- Training and development accuracy/perplexity plots saved as PNG files: attention_map_layer{i}_head{j}.png: Attention maps for layer i and head j.

Example Output

After running the task2 mode, you should see the training progress:

```yaml
Loading data and creating tokenizer ...
Vocabulary size is 5755
Iteration [100/500], Loss: 6.8656, Train Perplexity: 590.32
Test Perplexity on Obama: 732.09
Test Perplexity on H_Bush: 738.58
...
```
The attention maps will be saved in the current directory as attention_map_layer{i}_head{j}.png.