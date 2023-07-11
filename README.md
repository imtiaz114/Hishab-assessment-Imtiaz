# Hishab-assessment-Imtiaz

This repository contains my solution to the given assessment problem : Bangla person-name extractor.

I was given two separate publicly available dataset to finetune and produce result on. For this problem I have followed the following procedure:-

1. Preparing the datasets:

   This was done in the making_dataset.ipynb notebook. I followed these steps-
   
   a. All data was given token-tag pair in dictionary format, I formed sentences by identifying "|" which denoted the end of a sentence. Corresponding tags were also extracted so that token-tag pair relation reman intact.

   b. As I inteneded to use pretrain models and/or binary training approach I restricted number of labels to the most common ones, namely:
     {'O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC'}

   c. The other dataset had some annotation problem, like annotating ")" and "(" as separate token in some sentence and not in other, so after observing a bit, had to get rid of some of the token-tag pairs and had to remove all the punctuations from this dataset replacing with a single one in order to correctly tokenize the dataset.

3. Custom Approach :
   
  This approach is shown in the  [custom_model_clean_version.ipynb](https://github.com/imtiaz114/Hishab-assessment-Imtiaz/blob/main/clean_versions/custom_model_clean_version.ipynb) notebook. In this notebook basic preprocessing steps were handled as first like string to tokenization, aligning token tag pair, augmentation ( shuffling and random dropping), making lookup table, stratified splitting of dataset, building vocabulary to convert tokens into id. As for the architecture, I had start with conv-lstm architecture. But it was too simple and performed poorly. After that simple transformer based architecture was working but the results were still poor. After that conv1d and lstm layers were introduced in the architecture with a custom loss that suppressed the effect of padding, and "O" class and emphasized on "B-PER" and "I-PER" class. After a lot of experimentation a simple 2 transformer block followed by 2 transformer model has shown great results and it has  precision:  70.71%; recall:  88.43% on "PER" entity and a total accuracy of 93.47%. During testing some of the random sentences it had correctly extracted person name from all of them if present and didn't extract anything if not as expected. All of the coding was done in Tensorflow and transformer library was used. 
  
I have mainly used transformers for their superior performance in context extraction from sentences and superior memory than LSTM type recurrent architectures. Also, to stop them from overfitting the augmentations has performed well by increasing dataset size and also by changine sequence preventing models from giving results depending on token position in the sentence or any other dataset biases. The conv1d layers were introduced later to introduce local feature in the decision making along with transformers superior memory. It also serves purpose like - dimensionality reduction, complementary learning due to different strength in local and global feature extraction and acting as an additional feature extractor focusing mainly on local features to gather token level relation. Also, the custom loss played a huge roll in the model's success by guiding it towards optimizing for the "PER" entity.

![No Person case](https://github.com/imtiaz114/Hishab-assessment-Imtiaz/assets/83086464/a8d815e8-8c97-4ab7-9516-b445c509eed2)

A. No person name case.

![Multiple Person Case](https://github.com/imtiaz114/Hishab-assessment-Imtiaz/assets/83086464/985b4fbf-fcd7-47bc-9676-86515ab7b33f)

Models were trained in multi-class and binary manner, models trained on binary labels overfitted very quickly due to huge number of "O" entity in the dataset and very low number of "PER" entity. So, all experimentations and final models were done in the multi-class scnerio and later person name was extracted.

B. Multiple Person name case

3. Pretrained Models :

I have also experimented with pretrained models from Huggingface transformers. Among them some of which performed better are -

  a. "sagorsarker/bangla-bert-base"

  b. "sagorsarker/mbert-bengali-ner"

  c. "xlm-roberta-base"

  d. "ai4bharat/indic-bert"

  e. "Aleksandar/electra-srb-ner"

All of these models are NER related models. I have also tried base distillberts, roberta, bert, electra etc. but they didn't perform very good. The reason being short size of finetuning dataset. As a result my custom model performs better than these models in general but if dataset size is increased, I reckon this models or ensemble approach (majority voting) will be better performing. I have also used Tensorflow and transformer library for these models. I have tried both multi-class vs binary-classification problem sovling approach. As these models have more experience in language and NER task than my custom model, they perform significantly better in binary class scnerio as less distractions are present there. In multiclass scenerio they perform poorly and also the person entity are not kept intact. But binary class scnerio perform quite well. 

The clean version of these 2 approaches are given in [person-name-extractor-pt-bin.ipynb](https://github.com/imtiaz114/Hishab-assessment-Imtiaz/blob/main/clean_versions/person-name-extractor-pt-bin.ipynb) and [person-name-extractor-pt-bin.ipynb](https://github.com/imtiaz114/Hishab-assessment-Imtiaz/blob/main/clean_versions/person-name-extractor-pt-multi.ipynb) notebooks. The raw versions of all of the best result notebook are also provided in a separate folder. 

The performance of these models are presented in the  [performance comparison.txt](https://github.com/imtiaz114/Hishab-assessment-Imtiaz/blob/main/performance_comparison.txt) file. 










