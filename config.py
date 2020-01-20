from pathlib import Path
class Config:
    ROOT=Path().resolve()
    DATASETS=ROOT/ 'datasets'
    CC200_nyu=DATASETS
    Train_Info=DATASETS/'train_subjects_info.csv'
    Validate_Info=DATASETS/'validate_subjects_info.csv'
    Test_Info=DATASETS/'test_subjects_info.csv'
    All_Info=DATASETS/'ucla_subjects_info.csv'
