import keras
import os
import tensorflow as tf 

def train_model(dataset,model):
    
    with tf.device("/cpu:0"):
        model.compile(loss = keras.losses.binary_crossentropy,optimizer=keras.optimizers.Adam(1e-4),metrics=["accuracy"])
        model.fit_generator(dataset.train_generator(),epochs=30,steps_per_epoch=3000,validation_data=dataset.validation_generator(),validation_steps=len(dataset.validation),verbose=True)
        score = model.evaluate_generator(dataset.test_generator(),steps=len(dataset.test))
        print score
        with open("logs/log.txt","a+") as log_file:
            log_file.write("Score: "+str(score))
            log_file.write("\n")