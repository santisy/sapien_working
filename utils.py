import pandas as pd
import ast

def try_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val  # leave it as-is if not a Python literal

def read_metadata(csv_path):

    metadata = pd.read_csv(csv_path)

    # Apply to every column
    for col in metadata.columns:
        metadata[col] = metadata[col].apply(try_literal_eval)

    return metadata