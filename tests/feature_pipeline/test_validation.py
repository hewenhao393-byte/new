import pandas as pd
import pytest
from feature_pipeline.validation import ValidationError, validate_channel_alignment

def test_alignment_rejects_missing_channel_window():
    base=pd.DataFrame({"record_id":["r"],"split":["train"],"window_id":[0],"start_sample":[0],"end_sample":[4800]})
    with pytest.raises(ValidationError,match="alignment"):
        validate_channel_alignment({3:base,4:base,5:base.iloc[:0]})
