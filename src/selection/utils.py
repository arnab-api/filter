from src.models import unwrap_tokenizer


def get_first_token_id(name, tokenizer, prefix=" "):
    """Get the first token ID for a given name."""
    # print(f"{prefix=} | {name=}")
    return (
        tokenizer(prefix + name, return_tensors="pt", add_special_tokens=False)
        .input_ids[0][0]
        .item()
    )


class KeyedSet:
    def __init__(self, items, tokenizer):
        self.tokenizer = unwrap_tokenizer(tokenizer)
        self._dict = {get_first_token_id(item, tokenizer): item for item in items}

    def __sub__(self, other):
        diff_keys = set(self._dict.keys()) - set(other._dict.keys())
        values = [self._dict[k] for k in diff_keys]
        return KeyedSet(values, self.tokenizer)

    @property
    def keys(self):
        return list(self._dict.keys())

    @property
    def values(self):
        return list(self._dict.values())

    @property
    def len(self):
        return len(self._dict)

    def show(self):
        for k, v in self._dict.items():
            print(f'{k}["{self.tokenizer.decode(k)}"]: {v}')


# people_by_prof_set = {k: KeyedSet(v) for k, v in people_by_prof.items()}

# people_by_prof_set["actor"].show()
# print("-" * 50)
# people_by_prof_set["chef"].show()
# print("-" * 50)
# (people_by_prof_set["actor"] - people_by_prof_set["chef"]).show()
