library(keras)
library(tidyverse)
library(DataExplorer)
library(stringr)
library(caret)
library(reticulate)

data_full<-train
samp<-sample(nrow(data_full),nrow(data_full))
data_mixed<-data_full[samp,]
data_reduced<-data_mixed[1:100000,]
question_text<-(data_reduced$question_text)


targets<-c(data_reduced$target)

max_len<-100
max_words<-10000

glimpse(question_text)

tokenizer<-text_tokenizer(num_words = max_words,lower = TRUE,split=" ",char_level = FALSE)%>%
  fit_text_tokenizer(question_text)

sequences<-texts_to_sequences(tokenizer,question_text)

word_index=tokenizer$word_index

word_data<-pad_sequences(sequences,maxlen = max_len)

labels<-as.matrix(targets)

x_train<-word_data[1:50000,]
y_train<-labels[1:50000,]

x_val<-word_data[50001:75000,]
y_val<-labels[50001:75000,]

x_test<-word_data[75001:100000,]
y_test<-labels[75001:100000,]


model <- keras_model_sequential() %>%
  layer_embedding(input_dim = 10000,
                  output_dim = 8,
                  input_length = max_len) %>%
  layer_gru(units = 38,activation = "relu",recurrent_dropout = 0.1,
            return_sequences = T,dropout = 0.2) %>%
  layer_gru(units = 48,activation = "relu") %>%
  layer_dense(units = 64, activation = "relu")%>%
  layer_dense(units = 1, activation = "sigmoid")



model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)


history <- model %>% fit(
  x_train, y_train,
  epochs = 10,
  batch_size = 32,
  validation_split = 0.2,
  verbose=T
  #class_weight<-dict(list('0'=0.1,'1'=0.9))
)

model
plot(history)


