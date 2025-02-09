from pathlib import Path
from typing import Literal, get_args

import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, default_collate
from transformers import AutoTokenizer

NOTE_TYPE = Literal["hp", "op", "prog", "rad", "tert"]
DEFAULT_NOTE_TYPES = list(get_args(NOTE_TYPE))
DATA_CACHE = dict()


class TraumaDataset(Dataset):
    def __init__(
        self,
        *,  # enforce kwargs
        data_dir: str | Path,
        window: int,  # in hours
        note_types: list[NOTE_TYPE] = DEFAULT_NOTE_TYPES,
        tokenizer_name: str,
        context_length: int,
        n_splits: int = 5,
        include_splits: list[int] | None = None,
        exclude_splits: list[int] | None = None,
        split_seed: int = 42,
    ):
        if include_splits is not None and exclude_splits is not None:
            raise ValueError("Passing include and exclude split lists is redundant")
        super().__init__()
        if data_dir in DATA_CACHE:
            data = DATA_CACHE[data_dir]
        else:
            data = load_data(data_dir)
            DATA_CACHE[data_dir] = data
        patients = filter_patients(
            registry=data["registry"],
            demo=data["demo"],
            disch=data["disch"],
        )
        notes = filter_notes(
            patients=patients,
            window=window,
            note_types=note_types,
            **data,
        )
        joined_notes = join_notes(
            notes=notes,
            patients=patients,
        )
        labels, for_stratification = compute_labels(joined_notes)
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=split_seed,
        ).split(
            X=np.zeros(len(labels)),
            y=for_stratification,
        )
        idxs = []
        for i, (_, test_idxs) in enumerate(splitter):
            if include_splits is not None and i not in include_splits:
                continue
            if exclude_splits is not None and i in exclude_splits:
                continue
            idxs.append(test_idxs)
        idxs = np.concatenate(idxs)
        self.notes = joined_notes.loc[idxs].reset_index(drop=True)
        self.labels = labels.loc[idxs].reset_index(drop=True)
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name)
        self.context_length = context_length

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        x = self.notes.loc[i]
        label = self.labels.loc[i]
        out = self.tok(
            x["NOTE_TEXT"],
            truncation=True,
            padding="max_length",
            max_length=self.context_length,
            return_tensors="pt",
        )
        out = {k: v.squeeze() for k, v in out.items()}
        idx = min(out["attention_mask"].sum(), self.context_length - 1)
        out["input_ids"][idx] = self.tok.mask_token_id
        out["attention_mask"][idx] = 1
        for k, v in label.items():
            out[k] = torch.as_tensor(v)
        return out

    def __len__(self) -> int:
        return len(self.notes)


def compute_labels(metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    iss = pd.cut(metadata["ISS"], bins=[0, 25, 50, 75], labels=[0, 1, 2]).astype(int)
    mortality = (metadata["DC_DISPOSITION"] == "Expired").astype(int)

    labels = pd.DataFrame(
        {
            "mrn": metadata["MRN"],
            "hospital_mortality": mortality,
            "iss_tercile": iss,
        }
    )
    for_stratification = labels["hospital_mortality"].astype(str) + labels["iss_tercile"].astype(str)  # fmt: skip

    return labels, for_stratification


def join_notes(
    *,  # enforce kwargs,
    notes: pd.DataFrame,
    patients: pd.DataFrame,
) -> pd.DataFrame:
    joined_notes = (
        notes.sort_values(["MRN", "NOTE_DTTM", "SOURCE"])
        .groupby("MRN")["NOTE_TEXT"]
        .apply(lambda x: "\n".join(x))
        .reset_index()
    )
    return patients.merge(joined_notes, on="MRN")


def filter_notes(
    *,  # enforce kwargs
    patients: pd.DataFrame,
    window: int,
    note_types: list[NOTE_TYPE] = DEFAULT_NOTE_TYPES,
    **note_data,
) -> pd.DataFrame:
    note_lines = []
    for note_type in note_types:
        df = note_data[note_type]
        # different notes may come in at the same time but the same note should not have different timestamps
        assert (
            not df[["MRN", "NOTE_ID", "NOTE_DTTM"]]
            .drop_duplicates()[["MRN", "NOTE_ID"]]
            .duplicated()
            .any()
        )
        df["NOTE_DTTM"] = pd.to_datetime(df["NOTE_DTTM"])

        temp = patients.merge(df, on="MRN")
        window_notes = temp[
            (
                (temp["NOTE_DTTM"] >= temp["ED_ARRIVAL_DTTM"])
                & (
                    temp["NOTE_DTTM"]
                    < temp["ED_ARRIVAL_DTTM"] + pd.DateOffset(hours=window)
                )
            )
        ]
        note_lines.append(window_notes)
    note_lines = pd.concat(note_lines)
    # NOTE_ID should be unique across sources but may have multiple lines
    assert not note_lines[["NOTE_ID", "LINE"]].duplicated().any()

    # groupby preserves orders within each group so presort lines, group on note_id, and join lines
    joined_lines = (
        note_lines[["NOTE_ID", "LINE", "NOTE_TEXT"]]
        .sort_values(["NOTE_ID", "LINE"])
        .groupby("NOTE_ID")["NOTE_TEXT"]
        .apply(lambda x: "\n".join(x))
    )
    notes = (
        note_lines.drop_duplicates("NOTE_ID")
        .sort_values("NOTE_ID")
        .reset_index(drop=True)
    )
    assert (notes["NOTE_ID"] == joined_lines.index).all()
    notes["NOTE_TEXT"] = joined_lines.reset_index(drop=True)
    del notes["LINE"]
    return notes


def filter_patients(
    *,  # enforce kwargs
    registry: pd.DataFrame,
    demo: pd.DataFrame,
    disch: pd.DataFrame,
) -> pd.DataFrame:
    # in absence of unique stay identifier, filter to patients with only one stay
    registry = registry.drop_duplicates("Medical Record #", keep=False)
    demo = demo.drop_duplicates("MRN", keep=False).reset_index(drop=True)
    disch = disch.drop_duplicates("MRN", keep=False).reset_index(drop=True)

    registry = registry[registry["ISS"].str.isnumeric()]
    registry = registry.reset_index(drop=True)
    registry["ISS"] = registry["ISS"].astype(int)

    patients = disch.merge(demo, on="MRN")
    patients = patients.merge(registry, left_on="MRN", right_on="Medical Record #")

    # make sure ED arrival before hospital admit and discharge after arrival/admit
    patients = patients[patients["ED_ARRIVAL_DTTM"].notna()]
    patients = patients[
        (
            (patients["ED_ARRIVAL_DTTM"] <= patients["HSP_ADM_DTTM"])
            & (patients["HSP_DC_DTTM"] > patients["ED_ARRIVAL_DTTM"])
            & (patients["HSP_DC_DTTM"] > patients["HSP_ADM_DTTM"])
        )
    ]

    assert not patients["MRN"].duplicated().any()
    return patients[
        [
            "MRN",
            "ED_ARRIVAL_DTTM",
            "ED_DISPOSITION",
            "HSP_ADM_DTTM",
            "HSP_DC_DTTM",
            "DC_DISPOSITION",
            "DOB",
            "GENDER",
            "DOD",
            "DEATH_FLAG",
            "RACE",
            "ETHNICITY",
            "ISS",
        ]
    ]


def load_data(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)

    registry = pd.read_csv(data_dir / "Registry_Laparotomy.csv")
    demo = pd.read_csv(data_dir / "Demographics.csv")
    demo["DOB"] = pd.to_datetime(demo["DOB"])
    demo["DOD"] = pd.to_datetime(demo["DOD"])

    disch = pd.read_csv(data_dir / "DischargeDisposition.csv")
    disch["ED_ARRIVAL_DTTM"] = pd.to_datetime(disch["ED_ARRIVAL_DTTM"])
    disch["HSP_ADM_DTTM"] = pd.to_datetime(disch["HSP_ADM_DTTM"])
    disch["HSP_DC_DTTM"] = pd.to_datetime(disch["HSP_DC_DTTM"])

    prog = pd.read_csv(data_dir / "Progress_Notes.txt", sep="\t")
    op = pd.read_csv(data_dir / "OP_Notes.txt", sep="\t")
    tert = pd.read_csv(data_dir / "TertiarySurvery_Notes.txt", sep="\t")
    hp = pd.read_csv(data_dir / "HandP_Notes.txt", sep="\t")
    rad = pd.read_csv(data_dir / "Radiology_Imaging_Reports.txt", sep="\t")
    rad["LINE"] = 1
    rad["NOTE_TEXT"] = rad["RAD_NOTE"]

    prog["SOURCE"] = "prog"
    op["SOURCE"] = "op"
    tert["SOURCE"] = "tert"
    hp["SOURCE"] = "hp"
    rad["SOURCE"] = "rad"

    cols = ["MRN", "NOTE_ID", "NOTE_DTTM", "NOTE_TEXT", "LINE", "SOURCE"]
    prog = prog[cols]
    op = op[cols]
    tert = tert[cols]
    hp = hp[cols]
    rad = rad[cols]

    # radiology has some weird duplicates by note_id and differing datetimes
    # maybe a result of table joining?
    rad = rad.drop_duplicates().reset_index(drop=True)

    return {
        "registry": registry,
        "demo": demo,
        "disch": disch,
        "prog": prog,
        "op": op,
        "tert": tert,
        "hp": hp,
        "rad": rad,
    }


class LightningTraumaData(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        window: int,  # in hours
        tokenizer_name: str,
        context_length: int,
        batch_size: int,
        num_workers: int,
        val_split: int,
        test_split: int,
        n_splits: int = 5,
        split_seed: int = 42,
        note_types: list[NOTE_TYPE] = DEFAULT_NOTE_TYPES,
        inference_batch_size: int = 64,
        inference_num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str):
        if stage == "fit":
            self.train_ds = TraumaDataset(
                data_dir=self.hparams.data_dir,
                window=self.hparams.window,
                note_types=self.hparams.note_types,
                n_splits=self.hparams.n_splits,
                exclude_splits=[self.hparams.val_split, self.hparams.test_split],
                split_seed=self.hparams.split_seed,
                tokenizer_name=self.hparams.tokenizer_name,
                context_length=self.hparams.context_length,
            )
        if stage in ["fit", "validate"]:
            self.val_ds = TraumaDataset(
                data_dir=self.hparams.data_dir,
                window=self.hparams.window,
                note_types=self.hparams.note_types,
                n_splits=self.hparams.n_splits,
                include_splits=[self.hparams.val_split],
                split_seed=self.hparams.split_seed,
                tokenizer_name=self.hparams.tokenizer_name,
                context_length=self.hparams.context_length,
            )
        if stage in ["test", "predict"]:
            self.test_ds = TraumaDataset(
                data_dir=self.hparams.data_dir,
                window=self.hparams.window,
                note_types=self.hparams.note_types,
                n_splits=self.hparams.n_splits,
                include_splits=[self.hparams.test_split],
                split_seed=self.hparams.split_seed,
                tokenizer_name=self.hparams.tokenizer_name,
                context_length=self.hparams.context_length,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
            batch_size=self.hparams.batch_size,
            collate_fn=default_collate,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            batch_size=self.hparams.batch_size,
            collate_fn=default_collate,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            batch_size=self.hparams.inference_batch_size,
            collate_fn=default_collate,
            num_workers=self.hparams.inference_num_workers,
        )

    def predict_dataloader(self):
        return self.test_dataloader()
