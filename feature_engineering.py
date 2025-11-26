
import pandas as pd
import numpy as np

def basic_preprocess(df, is_train=True):
    df = df.copy()
    # Ensure Yr columns are ints
    for c in ["YearBuilt", "YearRemodAdd", "GarageYrBlt", "YrSold"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype('Int64')
    # Fill certain categorical NAs with "None"
    none_cols = [
        "Alley","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2",
        "FireplaceQu","GarageType","GarageFinish","GarageQual","GarageCond",
        "PoolQC","Fence","MiscFeature","MasVnrType"
    ]
    for c in none_cols:
        if c in df.columns:
            df[c] = df[c].fillna("None")
    # Numeric area columns where NA -> 0
    zero_cols = ["MasVnrArea","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF",
                 "GarageArea","GarageCars","PoolArea","WoodDeckSF","OpenPorchSF",
                 "EnclosedPorch","3SsnPorch","ScreenPorch","LowQualFinSF"]
    for c in zero_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    # LotFrontage: fill by median per Neighborhood
    if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
        df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
            lambda x: x.fillna(x.median())
        )
        df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())
    # Create engineered features
    if set(["GrLivArea","TotalBsmtSF"]).issubset(df.columns):
        df["TotalLivingArea"] = df["GrLivArea"] + df["TotalBsmtSF"]
    if "YearBuilt" in df.columns and "YearRemodAdd" in df.columns and "YrSold" in df.columns:
        df["AgeAtSale"] = df["YrSold"].astype("float") - df["YearBuilt"].astype("float")
        df["RemodAgeAtSale"] = df["YrSold"].astype("float") - df["YearRemodAdd"].astype("float")
    # Ordinal mapping
    qual_map = {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"None":0}
    for c in ["ExterQual","ExterCond","BsmtQual","BsmtCond","HeatingQC","KitchenQual","FireplaceQu","GarageQual","GarageCond","PoolQC"]:
        if c in df.columns:
            df[c+"_ord"] = df[c].map(qual_map).fillna(0).astype(int)
    return df

def encode_features(train, test, max_onehot=15):
    train = train.copy()
    test = test.copy()
    # Preserve target if present
    if "SalePrice" in train.columns:
        y = train["SalePrice"].copy()
        train = train.drop(columns=["SalePrice"])
    else:
        y = None
    # Identify categorical columns
    cat_cols = train.select_dtypes(include=["object","category"]).columns.tolist()
    num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    freq_encoded = []
    train_enc = train.copy()
    test_enc = test.copy()
    for c in cat_cols:
        n_unique = train[c].nunique()
        if n_unique <= max_onehot:
            continue
        else:
            freq = train[c].fillna("NA").value_counts(normalize=True)
            train_enc[c+"_freq"] = train[c].fillna("NA").map(freq)
            test_enc[c+"_freq"] = test[c].fillna("NA").map(freq).fillna(0)
            freq_encoded.append(c)
    for c in freq_encoded:
        train_enc = train_enc.drop(columns=[c])
        test_enc = test_enc.drop(columns=[c])
    combined = pd.concat([train_enc, test_enc], axis=0, sort=False)
    combined = pd.get_dummies(combined, drop_first=True)
    train_X = combined.iloc[:len(train_enc), :].copy()
    test_X = combined.iloc[len(train_enc):, :].copy().reset_index(drop=True)
    if y is not None:
        return train_X, test_X, y
    else:
        return train_X, test_X
