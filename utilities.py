
import matplotlib.pyplot as plt
import torch

class Utilities:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def sanity_check(self, sentence, block_size):
        # Encode the sentence using the tokenizer
        wordids = self.tokenizer.encode(sentence)

        # Prepare the padded input for the model
        padded_sentence = wordids[:block_size] + [0] * (block_size - len(wordids))
        input_tensor = torch.tensor(padded_sentence, dtype=torch.long).unsqueeze(0)
        
        # Move the input tensor to the same device as the model
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)

        # Create attention mask
        attention_mask = (input_tensor != 0).float().to(device)

        # Display input tensor shape
        print("Input tensor shape:", input_tensor.shape)

        # Process the input tensor through the encoder model with attention mask
        _, attn_maps = self.model(input_tensor, mask=attention_mask)

        # Display the number of attention maps (number of layers)
        print("Number of attention maps:", len(attn_maps))

        # Visualize and save the attention maps
        for j, attn_map in enumerate(attn_maps):
            att_map = attn_map.squeeze(0).detach()  # Remove batch dimension, shape: (n_head, seq_len, seq_len)
            n_head = att_map.shape[0]
            
            # Iterate over each attention head
            for h in range(n_head):
                head_attn = att_map[h]  # Shape: (seq_len, seq_len)
                
                # Check if the attention probabilities sum to 1 over rows
                total_prob_over_rows = torch.sum(head_attn, dim=1)
                if torch.any(total_prob_over_rows < 0.99) or torch.any(total_prob_over_rows > 1.01):
                    print(f"Failed normalization test in layer {j + 1}, head {h + 1}: probabilities do not sum to 1.0 over rows")
                    print("Total probability over rows:", total_prob_over_rows.cpu().numpy())
                
                # Move attention map to CPU and convert to NumPy array
                head_attn_numpy = head_attn.cpu().numpy()
                
                # Create a heatmap of the attention map
                fig, ax = plt.subplots()
                cax = ax.imshow(head_attn_numpy, cmap='Blues', interpolation='nearest')
                ax.xaxis.tick_top()
                fig.colorbar(cax, ax=ax)
                plt.title(f"Layer {j + 1}, Head {h + 1}")
                
                # Save the plot
                plt.savefig(f"attention_map_layer{j + 1}_head{h + 1}.png")
                
                # Show the plot
                plt.show()
            


