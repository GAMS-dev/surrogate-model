# Surrogate model example

This [GAMSPy](https://gamspy.readthedocs.io/en/latest/index.html) example shows how to embed a trained neural network into a GAMSPy optimization model (a mixed-integer linear program in this case) by utilizing the new sub-package [formulations](https://gamspy.readthedocs.io/en/latest/reference/gamspy.formulations.html) in GAMSPy.

More infos and examples on machine learning related capabilities of GAMSPy can be found in the GAMSPy user guide section on [GAMSPy and machine learning](https://gamspy.readthedocs.io/en/latest/user/ml/ml.html#gamspy-and-machine-learning).

## Quick start

If needed, setup a virtual environment or switch into a pre-existing one and install all packages from `requirements.txt`.

Spells to install from scratch with only a Python interpreter installed (tested with 3.12):
```
python -m venv venv
source venv/bin/activate # or use .ps1 script on Windows
pip install -r requirements.txt
python main.py
```

The NN is trained on the first run and saved to disk as `rs.pth`. Subsequent runs skip the training and only solve the optimization model.