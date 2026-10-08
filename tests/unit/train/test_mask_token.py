"""Training owns tokenizer metadata after preparation stops loading processors."""

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from speculators.train.utils import resolve_mask_token_id


@pytest.mark.parametrize("mode", ["existing", "add", "fallback"])
def test_resolve_mask_token_from_local_tokenizer(tmp_path, mode):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"<unk>": 0, "<eos>": 1, "x": 2})),
        unk_token="<unk>",
        eos_token="<eos>",
    )
    if mode == "existing":
        tokenizer.add_special_tokens({"mask_token": "<mask>"})
    tokenizer.save_pretrained(tmp_path)
    if mode == "fallback":
        with pytest.warns(UserWarning, match="pad_token_id=1"):
            assert resolve_mask_token_id(str(tmp_path), len(tokenizer)) == 1
    else:
        assert resolve_mask_token_id(str(tmp_path), len(tokenizer) + 1) == 3
