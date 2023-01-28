
# Get Intent from the text

`config.py` contains the parameters required for the model training/evaluation and predicting.

Feel free to tune the parameters and build the model accordingly using the following command

### Training

> python `bert_classify.py` --project bank --train


The model is generated under `models/bank` directory. Pick the top checkpoint and make sure "predict_params > model_checkpoint" parameter in the config.py is pointing to the path to the top checkpoint (or any other checkpoint you might want to use)

### Evaluate

> python `bert_classify.py` --project bank --predict --text "I want to transfer money to my another account"

or you can also send the text file (or any readable file) in which each line is an individual line for which you want to gather the intent

> python `bert_classify.py` --project bank --predict --file test.txt

