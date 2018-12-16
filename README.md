# unsupervised-RNN
The aim of this project is to train an RNN in an unsupervised fashion such that it can discover some clusters.
You will see the word 'sentiment' a lot in the code. It is because I'm hoping that either the network 'magically' 
discovers sentiment or I can come up with a better of discovering sentiment. The RNN is currently trained as a Language
Model on IMDB sentiment analysis dataset. Since I expect the sentiment to have a strong presence in this dataset, maybe 
the RNN can actually discover sentiment 'magically'.

The project is currently in experimentation stage.

### TO DO -- Programming part
- [ ] Refactor model.py -> return sentiments along with the next char prediction, make the whole hidden vairables thing
more clean, create a base class for the 2 classes but the sentiment block's gonna get kicked out anyway (see the 
research part below) and some other small refactorings
- [ ] Support more datasets
- [ ] Maybe support some command line arguments
- [ ] Add Tensorboard as PyTorch supports it now
- [ ] Try to find a library to print colourful text on an image(instead of console)

### TO DO -- Research part
- [ ] Replace sentiment block with "Multiscale hierarchical RNNs" to get word/phrase level clusters