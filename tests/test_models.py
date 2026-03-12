#%%
import torch
from src.transformer_modules.transformer_modules import TransformerEncoderModel
from src.modeles.transformer_classifier import TransformerClassifier
from src.modeles.rnn_classifier import RNNClassifier
from src.modeles.cnn_classifier import TextCNNClassifier


def _make_batch(batch_size=2, seq_len=8, vocab_size=50):
    torch.manual_seed(0)
    x = torch.randint(2, vocab_size, (batch_size, seq_len))
    # add some padding tokens
    x[0, -1] = 0
    x[1, -2:] = 0
    return x


def test_transformer_encoder_output_shape():
    x = _make_batch()
    mask = (x != 0)
    model = TransformerEncoderModel(
        vocab_size=50,
        d_model=32,
        N=2,
        h=4,
        d_ff=64,
        dropout=0.1,
    )

    out = model(x, mask)
    assert out.shape == (x.size(0), x.size(1), 32)


def test_transformer_classifier_output_shape():
    x = _make_batch()
    mask = (x != 0)
    encoder = TransformerEncoderModel(
        vocab_size=50,
        d_model=32,
        N=2,
        h=4,
        d_ff=64,
        dropout=0.1,
    )
    clf = TransformerClassifier(
        encoder,
        d_model=32,
        num_classes=1,
        pool_strategy="mean",
        dropout=0.1,
    )

    logits = clf(x, mask)
    assert logits.shape == (x.size(0), 1)


def test_rnn_classifier_output_shape():
    x = _make_batch()
    model = RNNClassifier(
        vocab_size=50,
        embed_dim=16,
        hidden_dim=32,
        num_layers=1,
        bidirectional=True,
        dropout=0.1,
        pad_idx=0,
        rnn_type="gru",
    )

    logits = model(x)
    assert logits.shape == (x.size(0),)


def test_textcnn_classifier_output_shape():
    x = _make_batch()
    model = TextCNNClassifier(
        vocab_size=50,
        embed_dim=16,
        num_filters=32,
        kernel_sizes=(3, 4, 5),
        dropout=0.1,
        pad_idx=0,
    )

    logits = model(x)
    assert logits.shape == (x.size(0),)

if __name__ == "__main__":
    test_transformer_encoder_output_shape()
    test_transformer_classifier_output_shape()
    test_rnn_classifier_output_shape()
    test_textcnn_classifier_output_shape()
    
    print("All tests passed!")    

# %%
