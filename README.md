This repository is for the course project of CS3507 Engineering Practice and Technological Innovation IV-J. I have implemented SVM model for a emotion recognition task over [SEED-IV](https://bcmi.sjtu.edu.cn/~seed/seed-iv.html) dataset.

---
## Install
```
pip install -r requirements.txt
```

## Run
```
python quad_svm_dependent.py
python quad_svm_independent.py
```
- Dependent train: depend on the subjects, which means we will train $15\times 3=45$ models for each subject each session.
- Independent train: independ from the subjects, which means we will train with data from $14$ subjects and test with another $1$ subject, and get $15$ models in total.