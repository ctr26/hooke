from dataclasses import dataclass
import polars as pl
import torch
import torch.nn.functional as F

# I add approximately 20% to the dimensions below to account for the new vocabulary items during fine-tuning
@dataclass(frozen=True)
class MetaDataConfig:
    rec_id_dim: int = 750_000 # Unique rec_ids:  628698
    concentration_dim: int = 550 # Unique concentrations:  386
    cell_type_dim: int = 55 # Unique cell_types:  47
    image_type_dim: int = 6 # Unique image_types:  4
    experiment_dim: int = 12_000 # Unique experiments:  10112
    well_address_dim: int = 1536 # Unique well_addresses:  1531


class Tokenizer:
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = []

    def fit(self, df: pl.DataFrame):
        unique_tokens = df.sort()
        for i, token in enumerate(unique_tokens):
            self.token_to_id[token] = i
            self.id_to_token.append(token)
        return self

    def transform(self, x):
        if isinstance(x, str):
            x = [x]
        return [self.token_to_id[token] for token in np.array(x).flatten()]

    def __len__(self):
        return len(self.id_to_token)

    def __call__(self, x):
        return self.transform(x)

    def state_dict(self) -> dict:
        return {"token_to_id": self.token_to_id, "id_to_token": self.id_to_token}

    @classmethod
    def from_state_dict(cls, state: dict) -> "Tokenizer":
        t = cls()
        t.token_to_id = state["token_to_id"]
        t.id_to_token = state["id_to_token"]
        return t


class DataFrameTokenizer:
    def __init__(self, df: pl.DataFrame, pad_length=8):
        self.rec_id_tokenizer = Tokenizer().fit(df["rec_id"].explode().unique())
        self.concentration_tokenizer = Tokenizer().fit(
            df["concentration"].explode().unique()
        )
        self.cell_type_tokenizer = Tokenizer().fit(df["cell_type"].unique())
        self.image_type_tokenizer = Tokenizer().fit(df["image_type"].unique())
        self.experiment_tokenizer = Tokenizer().fit(df["experiment_label"].unique())
        self.well_address_tokenizer = Tokenizer().fit(df["well_address"].unique())
        self.pad_length = pad_length

    def transform(self, row: dict[str, list[str]]):
        rec_id = self.rec_id_tokenizer(row["rec_id"])
        concentration = self.concentration_tokenizer(row["concentration"])

        return {
            "rec_id": F.pad(
                torch.tensor(rec_id, dtype=torch.long),
                (0, self.pad_length - len(rec_id)),
            ),
            "concentration": F.pad(
                torch.tensor(concentration, dtype=torch.long),
                (0, self.pad_length - len(concentration)),
            ),
            "comp_mask": F.pad(
                torch.ones(len(rec_id), dtype=torch.long),
                (0, self.pad_length - len(rec_id)),
            ).to(torch.bool),
            "cell_type": torch.tensor(
                self.cell_type_tokenizer(row["cell_type"])[0], dtype=torch.long
            ),
            "image_type": torch.tensor(
                self.image_type_tokenizer(row["image_type"])[0], dtype=torch.long
            ),
            "experiment_label": torch.tensor(
                self.experiment_tokenizer(row["experiment_label"])[0], dtype=torch.long
            ),
            "well_address": torch.tensor(
                self.well_address_tokenizer(row["well_address"])[0], dtype=torch.long
            ),
        }

    def __call__(self, row):
        return self.transform(row)

    def state_dict(self) -> dict:
        return {
            "rec_id": self.rec_id_tokenizer.state_dict(),
            "concentration": self.concentration_tokenizer.state_dict(),
            "cell_type": self.cell_type_tokenizer.state_dict(),
            "image_type": self.image_type_tokenizer.state_dict(),
            "experiment": self.experiment_tokenizer.state_dict(),
            "well_address": self.well_address_tokenizer.state_dict(),
            "pad_length": self.pad_length,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "DataFrameTokenizer":
        t = object.__new__(cls)
        t.rec_id_tokenizer = Tokenizer.from_state_dict(state["rec_id"])
        t.concentration_tokenizer = Tokenizer.from_state_dict(state["concentration"])
        t.cell_type_tokenizer = Tokenizer.from_state_dict(state["cell_type"])
        t.image_type_tokenizer = Tokenizer.from_state_dict(state["image_type"])
        t.experiment_tokenizer = Tokenizer.from_state_dict(state["experiment"])
        t.well_address_tokenizer = Tokenizer.from_state_dict(state["well_address"])
        t.pad_length = state["pad_length"]
        return t
