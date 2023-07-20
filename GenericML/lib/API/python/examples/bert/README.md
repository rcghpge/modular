# BERT example

Assuming you already have the `modular` Python package installed, you now need
a couple more packages you can get with this command, run in this directory:

```sh
python3 -m pip install -r requirements.txt
```

Then download the BERT model from Hugging Face:

```sh
python3 download-model.py
```

Now you can run an inference (using dummy data):

```sh
python3 main.py
```
