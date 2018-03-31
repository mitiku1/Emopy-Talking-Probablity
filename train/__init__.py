import keras
import os
import tensorflow as tf 

def train_model(dataset,model):
    
    with tf.device("/cpu:0"):
        model.summary()
        model.compile(loss = keras.losses.binary_crossentropy,optimizer=keras.optimizers.Adam(1e-4),metrics=["accuracy"])
        model.fit_generator(dataset.train_generator(model),epochs=30,steps_per_epoch=3000,validation_data=dataset.validation_generator(model),validation_steps=len(dataset.validation)*3,verbose=True)
        model.save_weights("models/model.h5")
        score = model.evaluate_generator(dataset.test_generator(model),steps=len(dataset.test)*3)
        print score
        with open("logs/log.txt","a+") as log_file:
            log_file.write("Score: "+str(score))
            log_file.write("\n")