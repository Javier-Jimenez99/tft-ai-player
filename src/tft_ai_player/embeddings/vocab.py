"""Universal Vocabularies for Champion IDs, Item IDs, Traits, and Aliases."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np


class ChampionVocabulary:
    """Universal vocabulary mapping discrete champion strings to deterministic IDs."""

    def __init__(self, champions: Sequence[str] | None = None) -> None:
        self.champ_to_idx: dict[str, int] = {}
        self.idx_to_champ: dict[int, str] = {0: "<EMPTY>"}

        self._initialize_defaults()

        if champions:
            for champ in champions:
                self.add_champion(champ)

    def _initialize_defaults(self) -> None:
        """Populate initial vocabulary from canonical Set 18 champion catalog."""
        try:
            from tft_ai_player.simulation.sets.set18 import SET18_CHAMPION_CATALOG
            for c in SET18_CHAMPION_CATALOG:
                self.add_champion(c.champion_id)
        except Exception:
            pass

    def add_champion(self, name: str) -> int:
        if not name or name == "<EMPTY>":
            return 0
        norm_name = self.normalize_name(name)
        if norm_name in self.champ_to_idx:
            return self.champ_to_idx[norm_name]

        new_idx = len(self.champ_to_idx) + 1
        self.champ_to_idx[norm_name] = new_idx
        self.idx_to_champ[new_idx] = norm_name

        if name != norm_name and name not in self.champ_to_idx:
            self.champ_to_idx[name] = new_idx

        return new_idx

    @staticmethod
    def normalize_name(name: str) -> str:
        if not name:
            return "<EMPTY>"
        return name.strip()

    def encode(self, name: str | None) -> int:
        if not name:
            return 0
        norm = self.normalize_name(name)
        if norm in self.champ_to_idx:
            return self.champ_to_idx[norm]
        if name in self.champ_to_idx:
            return self.champ_to_idx[name]
        return self.add_champion(name)

    def decode(self, idx: int) -> str:
        return self.idx_to_champ.get(idx, "<EMPTY>")

    def lookup(self, key: int | str) -> str | int:
        if isinstance(key, (int, np.integer)):
            return self.decode(int(key))
        return self.encode(str(key))

    def __len__(self) -> int:
        return len(self.idx_to_champ)

    def to_dict(self) -> dict[str, int]:
        return dict(self.champ_to_idx)

    def save(self, path: str | Path) -> None:
        data = {
            "champ_to_idx": self.champ_to_idx,
            "idx_to_champ": {str(k): v for k, v in self.idx_to_champ.items()},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> ChampionVocabulary:
        vocab = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab.champ_to_idx = data.get("champ_to_idx", {})
        vocab.idx_to_champ = {int(k): v for k, v in data.get("idx_to_champ", {}).items()}
        return vocab


class ItemVocabulary:
    """Universal vocabulary mapping discrete item strings to deterministic IDs."""

    def __init__(self, items: Sequence[str] | None = None) -> None:
        self.item_to_idx: dict[str, int] = {}
        self.idx_to_item: dict[int, str] = {0: "<NO_ITEM>"}

        self._initialize_defaults()

        if items:
            for item in items:
                self.add_item(item)

    def _initialize_defaults(self) -> None:
        try:
            from tft_ai_player.simulation.sets.set18 import SET18_ITEMS_CATALOG
            for item_id in SET18_ITEMS_CATALOG:
                self.add_item(item_id)
        except Exception:
            pass

    def add_item(self, name: str) -> int:
        if not name or name == "<NO_ITEM>":
            return 0
        norm_name = name.strip()
        if norm_name in self.item_to_idx:
            return self.item_to_idx[norm_name]

        new_idx = len(self.item_to_idx) + 1
        self.item_to_idx[norm_name] = new_idx
        self.idx_to_item[new_idx] = norm_name

        return new_idx

    def encode(self, name: str | None) -> int:
        if not name:
            return 0
        norm = name.strip()
        if norm in self.item_to_idx:
            return self.item_to_idx[norm]
        return self.add_item(name)

    def decode(self, idx: int) -> str:
        return self.idx_to_item.get(idx, "<NO_ITEM>")

    def lookup(self, key: int | str) -> str | int:
        if isinstance(key, (int, np.integer)):
            return self.decode(int(key))
        return self.encode(str(key))

    def __len__(self) -> int:
        return len(self.idx_to_item)

    def to_dict(self) -> dict[str, int]:
        return dict(self.item_to_idx)

    def save(self, path: str | Path) -> None:
        data = {
            "item_to_idx": self.item_to_idx,
            "idx_to_champ": {str(k): v for k, v in self.idx_to_item.items()},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> ItemVocabulary:
        vocab = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab.item_to_idx = data.get("item_to_idx", {})
        vocab.idx_to_item = {int(k): v for k, v in data.get("idx_to_champ", {}).items()}
        return vocab


class TraitVocabulary:
    """Universal vocabulary and synergy counter for traits in active sets."""

    def __init__(self, traits: Sequence[str] | None = None) -> None:
        self.trait_to_idx: dict[str, int] = {}
        self.idx_to_trait: dict[int, str] = {}
        self.champ_trait_map: dict[str, list[str]] = {}

        self._initialize_defaults()

        if traits:
            for trait in traits:
                self.add_trait(trait)

    def _initialize_defaults(self) -> None:
        try:
            from tft_ai_player.simulation.sets.set18.profile import SET18_CHAMPION_CATALOG, SET18_TRAIT_CATALOG
            for trait_name in SET18_TRAIT_CATALOG:
                self.add_trait(trait_name)
            for c_id, c in SET18_CHAMPION_CATALOG.items():
                self.register_champ_traits(c_id, c.traits)
                self.register_champ_traits(c.name, c.traits)
                clean_name = c_id.replace("TFT18_", "")
                self.register_champ_traits(clean_name, c.traits)
                self.register_champ_traits(clean_name.lower(), c.traits)
                self.register_champ_traits(f"DA_18_{clean_name}", c.traits)
                self.register_champ_traits(f"DA_{clean_name}18", c.traits)
                self.register_champ_traits(f"DA_{clean_name}", c.traits)
        except Exception:
            pass

    def add_trait(self, name: str) -> int:
        if not name:
            return 0
        norm_name = name.strip()
        if norm_name in self.trait_to_idx:
            return self.trait_to_idx[norm_name]

        new_idx = len(self.trait_to_idx)
        self.trait_to_idx[norm_name] = new_idx
        self.idx_to_trait[new_idx] = norm_name
        return new_idx

    def encode(self, name: str | None) -> int:
        """Encode trait string to numeric integer index."""
        if not name:
            return 0
        return self.add_trait(name)

    def lookup(self, key: int | str) -> str | int:
        if isinstance(key, (int, np.integer)):
            return self.idx_to_trait.get(int(key), "")
        return self.trait_to_idx.get(str(key), 0)


    def __len__(self) -> int:
        return len(self.trait_to_idx)

    def register_champ_traits(self, champ_name: str, traits: Sequence[str]) -> None:
        if not champ_name:
            return
        norm_champ = champ_name.strip()
        self.champ_trait_map[norm_champ] = [t.strip() for t in traits if t]
        for t in traits:
            if t:
                self.add_trait(t)

    def _get_traits_for_champ(self, champ_name: str) -> list[str]:
        c = champ_name.strip()
        if c in self.champ_trait_map:
            return self.champ_trait_map[c]
        # Try stripping common prefixes and suffixes
        clean = re.sub(r"^(TFT18_|DA_18_|DA_)", "", c, flags=re.IGNORECASE)
        clean = re.sub(r"(18|_AP|_Small|_small)$", "", clean)
        if clean in self.champ_trait_map:
            return self.champ_trait_map[clean]
        if clean.lower() in self.champ_trait_map:
            return self.champ_trait_map[clean.lower()]
        return []

    def compute_trait_vector(self, champ_names: Sequence[str]) -> np.ndarray:
        """Compute active trait synergy counts vector across unique board champions."""
        vec = np.zeros(max(len(self.trait_to_idx), 1), dtype=np.float32)
        unique_champs = set(c.strip() for c in champ_names if c and c.strip() != "<EMPTY>")

        trait_counts: Counter[str] = Counter()
        for c in unique_champs:
            for t in self._get_traits_for_champ(c):
                trait_counts[t] += 1

        for trait, count in trait_counts.items():
            if trait in self.trait_to_idx:
                idx = self.trait_to_idx[trait]
                vec[idx] = float(count)

        return vec

    def get_active_synergies(self, champ_names: Sequence[str]) -> dict[str, int]:
        """Compute active trait synergy count dictionary across unique board champions."""
        unique_champs = set(c.strip() for c in champ_names if c and c.strip() != "<EMPTY>")
        trait_counts: Counter[str] = Counter()
        for c in unique_champs:
            for t in self._get_traits_for_champ(c):
                trait_counts[t] += 1
        return dict(trait_counts)



    def __len__(self) -> int:
        return len(self.trait_to_idx)

    def to_dict(self) -> dict[str, int]:
        return dict(self.trait_to_idx)

    def save(self, path: str | Path) -> None:
        data = {
            "trait_to_idx": self.trait_to_idx,
            "idx_to_trait": {str(k): v for k, v in self.idx_to_trait.items()},
            "champ_trait_map": self.champ_trait_map,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> TraitVocabulary:
        vocab = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab.trait_to_idx = data.get("trait_to_idx", {})
        vocab.idx_to_trait = {int(k): v for k, v in data.get("idx_to_trait", {}).items()}
        vocab.champ_trait_map = data.get("champ_trait_map", {})
        return vocab
