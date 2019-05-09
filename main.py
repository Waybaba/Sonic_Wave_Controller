import config
from Data_Processing.data_processing import get_data
import time

import model.fc_cnn_model as fc_model
import model.unet_raw as unet_raw_model
from keras.callbacks import TensorBoard

# get a special name
run_name = "fc_cnn_first"
special_name = "0509_"+time.asctime(time.localtime(time.time())).replace(" ", "_")[-12:-7] + run_name

# wrapping all the data pre-processing steps in the get_data() function
train_data,train_label,test_data,test_label = get_data()



# wrapping all the model bulding steps in the get_model() function
model = fc_model.get_model()
model.summary()

# train
model.fit(
        train_data,train_label,
        epochs=100,
        batch_size=4,
        verbose=1,
        callbacks= [TensorBoard(log_dir=config.project_path + 'Others/logs/' +special_name+"/")],
        validation_data=(test_data,test_label)
    )


"""




"""