from pathlib import Path
import pickle
import rdkit
import torchdrug
import warnings

from rdkit import Chem

# orderd by perodic table
atom_vocab = [
    "H",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Mg",
    "Si",
    "P",
    "S",
    "Cl",
    "Cu",
    "Zn",
    "Se",
    "Br",
    "Sn",
    "I",
]
atom_vocab = {a: i for i, a in enumerate(atom_vocab)}

chiral_tag_vocab = range(4)


def read_pickle(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def onehot(x, vocab, allow_unknown=False):
    if x in vocab:
        if isinstance(vocab, dict):
            index = vocab[x]
        else:
            index = vocab.index(x)
    else:
        index = -1
    if allow_unknown:
        feature = [0] * (len(vocab) + 1)
        if index == -1:
            warnings.warn("Unknown value `%s`" % x)
        feature[index] = 1
    else:
        feature = [0] * len(vocab)
        if index == -1:
            raise ValueError(
                "Unknown value `%s`. Available vocabulary is `%s`" % (x, vocab)
            )
        feature[index] = 1

    return feature


def atom_pretrain(atom):
    """Atom feature for pretraining.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol

        GetChiralTag(): one-hot embedding for atomic chiral tag
    """
    onehot_atom = onehot(atom.GetSymbol(), atom_vocab)
    onehot_chiral = onehot(atom.GetChiralTag(), chiral_tag_vocab)
    return onehot_atom + onehot_chiral


if __name__ == "__main__":
    print(f"rdkit_version:{rdkit.__version__}")
    print(f"torchdrug_version:{torchdrug.__version__}")

    # Create a molecule consisting of a single carbon atom
    carbon_atom = Chem.Atom(6)

    # Print the molecule to ensure it's correctly defined

    s = atom_pretrain(carbon_atom)
    print(len(s))

    data_path = Path("data/processed/sample/Q18820.pkl")
    protein = read_pickle(data_path)
    print(protein.atom_feature.shape[1])
    print(protein)
