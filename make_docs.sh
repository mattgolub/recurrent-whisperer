# Currently running this script fails for all of the pydoc commands,
# but all commands work if copy and pasted into the command line.
pydoc -w ./RecurrentWhisperer.py
pydoc -w ./Hyperparameters.py
pydoc -w ./AdaptiveLearningRate.py
pydoc -w ./AdaptiveGradNormClip.py
pydoc -w ./Timer.py
mv *.html ./documentation