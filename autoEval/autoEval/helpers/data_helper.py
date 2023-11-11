import pandas as pd


def detect_categorical_columns(df, threshold=0.05):
    categorical_cols = []
    for col in df.columns:
        unique_values = df[col].nunique()
        total_values = df[col].count()
        if unique_values / total_values < threshold:
            categorical_cols.append(col)
    return categorical_cols


def flatten(dict_result):
    def flatten_dict(d, parent_key="", sep="_"):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    flat_data = flatten_dict(dict_result)

    # Convert to DataFrame
    df = pd.DataFrame(flat_data, index=[0])

    # Transpose the DataFrame
    df = df.T.reset_index()

    # Rename columns
    df.columns = ["Metric", "Score"]

    df.sort_values(by="Metric")
    return df
