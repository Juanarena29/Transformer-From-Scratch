"""Config.py
Transformer hyperparameter container.
Architecture position: consumed before model construction and used by training
and inference scripts after tokenizer loading.
"""


class TransformerConfig:
    """Static hyperparameter configuration for the Transformer model.

    Returns
    -------
    None
        This class only exposes class attributes used as defaults.
    """

    vocab_size = 6000  # Total tokenizer vocabulary size.
    d_model = 128  # Embedding and hidden representation dimensionality.
    n_heads = 8  # Number of attention heads per encoder block.
    n_layers = 4  # Number of stacked encoder blocks.
    max_seq_len = 512  # Maximum sequence length supported by the model.
    d_ff = 512  # Feed-forward hidden dimensionality inside each block.
    lr = 0.001  # Default learning rate for SGD updates.
