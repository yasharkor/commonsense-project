

# Commonsense Validation

## Introduction

The research project is to directly test whether a system can differentiate natural language statements that make sense from those that do not make sense. This project is designed based on SemEval2020 task 4 subtaskA. The task is to choose from two natural language statements with similar wordings which one makes sense and which one does not make sense.

This project is based on the SemEval Task 4-A: Commonsense Validation and Explanation.This project used the organizer’s data to imple-ment  various  models  to  distinguish  commonsense  statements  versus  non-common  sensestatements.  Pre-trained language models suchas  Bert  and  Gpt2  and  random  selection  are utilized  as  baselines.Furthermore,  we  implemented  and fine-tuned  state-of-the-art lan-guage  models  such  as  Bert to enhance common-sense prediction performance further.  Besides,  we utilized statistical significance tests to ensure the reliability of the obtained results.


### Example

#### Task A: Commonsense Validation

Which statement of the two is against common sense?

- Statement 1: He put a turkey into the fridge. *(correct)*
- Statement 2: He put an elephant into the fridge.




## Evaluation

Subtask A will be evaluated using **accuracy**

## Data format

The file format is csv.

Each row of the csv file contains 3 fields: `id`, `sent0`, `sent1`, which are the ID of the instance, and two input sentences. The output file contains no header, and each row contains the id and the index of the sentence which makes sense.

The data-set is divided into three subsets, including training, development, and test set. Training setincludes 10,000 samples each of them composedof two statements i.e.s1,s2.  Therefore, 10,000sentences make sense, and there are 10,000 sen-tences against common sense.  The developmentset includes 997,  and the test set includes 1000samples.  The training, development and test setsare provided by the SemEval2020 organizer in aCSV format.


## Installation and Execution Instructions
- Clone the repository
- Navigate to the code folder of the repository (```project-yasharkor\code``` on github)
- from the terminal execute the following command: ``` python3 .\main.py [mode] ```
- The **[training mode]** argument is the mode you wish to run the program in. You can select from the following modes:

  - **bert** - BERT language model for perplexity calculation and commonsense prediction on Test data
  - **gpt2** - GPT2 language model for perplexity calculation and commonsense prediction on Test data
  - **finebert** - Fine tunning bert calssifier model on Training data and evaluating on Dev data and final test on Test data
- An example usage of this program would be: ``` python3 ./main.py gpt2```


## Results
- GPT2 Accuracy is equal to: 67.5 %
- Bert Accuracy is equal to: 54.80 %
- Fine tuned bert Accuracy on evaluation set On Test set Total MCC: 0.554, Accuracy: 0.75


|       |Training Loss|  Valid. Loss|  Valid. Accur.| Training Time| Validation Time|
|-------|-------------|-------------|---------------|---------------|---------------|
|epoch  |             |             |               |               |               |
|1      |0.59         |0.55         |0.72           |0:30:34        |0:00:26        |
|2      |0.38         |0.56         |0.73           |0:28:59        |0:01:10        |
|3      |0.30         |0.56         |0.73           |0:33:00        |0:00:29        |
|4      |0.30         |0.56         |0.73           |0:28:38        |0:00:29        |





## Sources 
https://github.com/wangcunxiang/SemEval2020-Task4-Commonsense-Validation-and-Explanation
https://colab.research.google.com/drive/1pTuQhug6Dhl9XalKB0zUGf4FIdYFlpcX#scrollTo=nskPzUM084zL
https://medium.com/@aksun/i-put-an-elephant-in-my-fridge-using-nlp-techniques-to-recognize-the-absurdity-2d8d565659e
https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128
https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification

