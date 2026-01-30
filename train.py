import sys, os
import time
import pandas as pd
import logging
from units.models import ModelTrainer
from units.unit import load_data
from sklearn.preprocessing import StandardScaler, MinMaxScaler

if __name__ == "__main__":
    start_time = time.time()

    y_type = 'dG_kcal_mol'

    local_descriptors = pd.read_csv('all_descriptor.csv',index_col='name')
    local_descriptors = local_descriptors.drop(columns=['HOMO','HOMO_number', 'LUMO', 'HOMO_LUMO_Gap'])
    combined_descriptors = pd.concat([local_descriptors], axis=1)

    data = combined_descriptors
    data_y = pd.read_csv('dG_ddG_PXP_cat_ts.csv', index_col=0)

    data = data.loc[data_y.index]

    y = data_y[y_type]

    nan_columns = data.columns[data.isnull().any()].tolist()
    constant_columns = data.columns[data.nunique() == 1].tolist()

    columns_to_drop = list(set(nan_columns + constant_columns))
    if columns_to_drop:

        print(f"⚠️ Warning: Found {len(columns_to_drop)} problematic feature columns that need to be removed.")

        if nan_columns:
            print(f"  - {len(nan_columns)} columns contain NaN values: {nan_columns}")
        if constant_columns:
            print(f"  - {len(constant_columns)} columns are constant: {constant_columns}")
        
        data = data.drop(columns=columns_to_drop)

    cols = data.columns

    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(data), 
                    index=y.index, 
                    columns=cols)
    corr = X.corr().abs()
    keep = []
    for i in range(len(corr.columns)):
        above = corr.iloc[:i,i]
        if keep: above = above[keep]
        if all(above < 0.95):
            keep.append(corr.columns[i])
    X = X[keep]
    if y_type == 'ddG_kcal_mol':
        X = X[
                [
            'c_CM5_Au2',
            'Farthest_Distance',
            'Mol_Size_Short',
            'Bond_Au_C',
            'Angle_C_Au_I',
            'bv_P_1',
            'bv_P_2',
            'pyr_p_Au_1',
            'sasa_P_1',
            'sasa_P_2'
                ]
            ]
    elif y_type == 'dG_kcal_mol':
        X = X[
                [
            'c_Mulli_P1',
            'LEA_Ave',
            'Bond_Au_I',
            'Angle_C_Au_I',
            'sasa_P_2',
            'l_shell_d',
            'Pop_type_s_I',
            'Pop_type_p_I',
            'Atom_d_m_Z_Au_1',
            'd_m_X'
                ]
            ]

    print(f"Data shape after removing highly correlated features: {X.shape}")

    results_path = f'10_feature_results_{y_type}'
    os.makedirs(results_path, exist_ok=True)

    trainer = ModelTrainer(X, y, results_path)
    trainer.train_linear_models()
    trainer.train_decision_tree()
    trainer.train_svr()
    trainer.train_xgboost()
    trainer.train_random_forest()
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal time: {total_time:.2f} seconds (~{total_time/60:.2f} minutes)")

