def detect_categorical_columns(df, threshold=0.05):
    categorical_cols = []
    for col in df.columns:
        unique_values = df[col].nunique()
        total_values = df[col].count()
        if unique_values / total_values < threshold:
            categorical_cols.append(col)
    return categorical_cols
