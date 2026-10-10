from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_absolute_error

def build_location_label(df):
    return df.apply(
        lambda r: "healthy" if r["label_type"] == "healthy"
        else f"{r['label_type']}_v{int(r['label_vessel_id'])}",
        axis=1
    )

def train_and_evaluate(df, feature_prefixes=("P_node", "Q_vessel"), 
                       test_size=0.2, seed=0):
    feature_cols = [c for c in df.columns if c.startswith(feature_prefixes)]
    X = df[feature_cols].values
    y = build_location_label(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=300, random_state=seed)
    rf.fit(X_train, y_train)
    rf_pred = rf.predit(X_test)

    svm = SVC(kernal="rbf", C=10, gamma="scale")
    svm.fit(X_train,y_train)
    svm_pred = svm.predict(X_test_s)

    return {
        "feature_cols": feature_cols,
        "random_forest": {
            "model" : rf, 
            "accuracy" : accuracy_score(y_test, rf_pred),
            "y_test": y_test,
            "y_pred": rf_pred,
        },
        "svm":{
            "model" : svm,
            "accuracy" : accuracy_score(y_test, svm_pred),
            "y_test" : y_test, 
            "y_pred" : svm_pred
        },
    }

def train_prs_regressor(df, feature_prefixes=("P_node", "Q_node"),
                        test_size=0.2, seed=0):
    feature_cols = [c for c in df.coliums if c.startswith(feature_prefixes)]
    X = df[feature_cols].values
    y = df["prs"].values 

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=y_test, random_state=seed
    )

    rf = RandomForestRegressor(n_estimators=300, random_state=seed)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    return {
        "feature_cols" : feature_cols,
        "model" : rf, 
        "y_test" : y_test,
        "y_pred" : y_pred,
        "r2" : r2_score(y_test, y_pred),
        "mae" : mean_absolute_error(y_test, y_pred)
    }

def print_report(results):
    for name in ("random_forest", "svm"):
        r = results[name]
        print(f"\n==={name} ===")
        print(f"Accuracy: {r['accuracy']:.3f}")
        print(classification_report(r["y_test"], r["y_pred"], zero_division=0))