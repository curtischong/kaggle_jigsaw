# kaggle_jigsaw

Written by [Kevin Shen](https://github.com/kshen3778), [Chris Wang](https://github.com/Christopher-Wang), and [Curtis Chong](https://github.com/curtischong)

[Link to the Competition](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/overview)

### Premise
Many text-based models usually suffer from bias towards certain words.

For example, a model that classifies offensive remarks may mistake the phrase “I am a gay woman” as offensive because the word “gay” biases the model into thinking that the sentence is offensive.

To further research in this domain, Jigsaw, a subsidiary of Alphabet, challenged Kaggle to remove bias from models that detect toxicity within online comments.

### Approach

We started with LSTMs and pre-processed by removing emoticons and special symbols. Since we wanted to maximize our coverage on the Glove and FastText embeddings, we cleaned the text with common phrases from a communal dictionary. During this first month, we also implemented stratified k-fold (a million times too many), sped up training with sequence bucketing, and tinkered around with GRUs.

Finding little success with LSTMs, we moved on to fine-tune BERT, a transformer language model that excelled at handling un-preprocessed text (rip all the efforts we put into cleaning). In order to get our comments to fit within the GPU and memory limit, we truncated comments that were too long. For every comment that exceeded 360 characters, we took the first 90 chars and last 270 chars of the message and disregarded the characters in the middle because EDA showed that people tended to get toxic at the end of their comments.

Next, we implemented mini batching to speed the training of BERT and ran it for 2 epochs. At this point, each model took 14 hours to train and we wanted to ensemble 5 of them.... With a week left, we hedged our bets on linearly discriminating learning rates, and a search for the perfect amount of epochs.

Another thing we tried was GPT-2 and Bert Large, but we didn’t have the compute power or enough data to make it work. Let’s not even mention XLNet being published 1 week before the deadline. So much stress.


### The Final Stretch

After an accidental $500 fee racked up on our cloud GPU and 3 months of sleepless nights, we bagged our 5 best models and finished with a score of 0.94187, placing us in 195th place out of 3,165 teams.

And that was how we spent our summer :) 
