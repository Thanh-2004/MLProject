# This file use to record data.
import numpy as np
import pickle
import serial
from Preprocessing import slide_func, filter_data, FeatureExtract
from PIL import Image, ImageTk
import tkinter as tk
import time
import pandas as pd
if serial.Serial:
    serial.Serial().close()

# Open the serial port
# s = serial.Serial("/dev/tty.usbmodem1301", baudrate=57600)  # COMx in window or /dev/ttyACMx in Ubuntu with x is number of serial port.
loaded_model = pickle.load(open("SourceCode/ModelAI/trained_model/RandomForestClassifier.h5", 'rb'))
print("START!")


def start_test():
    x = 0
    y = np.array([], dtype=int)
    k = 20  # window
    sample_rate = 512
    window_size = k * sample_rate
    feature = []
    path = "/Users/nguyentrithanh/Documents/20232/MachineLearning/CapstoneProjectML/CollectedData/new_data/ThanhfVui.txt"
    file = open(path, "r")
    while x < (1000 * sample_rate):
        try:
            if x % sample_rate == 0:
                print(x // sample_rate)
            # data = file.readline().decode('utf-8').rstrip("\r\n")
            data = file.readline()
            y = np.append(y, int(data))
            x += 1
            if x >= window_size:
                if x % (1 * sample_rate) == 0:
                    sliding_window_start = x - window_size
                    sliding_window_end = x
                    slide = np.array(y[sliding_window_start:sliding_window_end])  # sliding_window ~ ys
                    slide = filter_data(slide)
                    feature_window = FeatureExtract(slide, plot=1)
                    feature.append(feature_window)
                    test_slide = pd.DataFrame.from_dict(FeatureExtract(slide, plot=0)).values
                    predictions = loaded_model.predict(test_slide)
                    print(predictions)
                    print(1-np.count_nonzero(predictions)/15)
                # print(int(data))
                # file.write(data)
                # file.write('\n')
                    show_image()
                    time.sleep(0.5)
        except:
            pass
    # np.savetxt("D:\Tuda\Research\Data_and_AI\Data_new\Tuda_00000.txt", y, fmt="%d")  # Save in int


# Use to get data txt


image_path1 = '/Users/nguyentrithanh/Documents/20232/MachineLearning/CapstoneProjectML/SourceCode/test.png'



def show_image():
    image1 = Image.open(image_path1)
    photo1 = ImageTk.PhotoImage(image1)
    image_frame1.configure(image=photo1)
    window.update()
    image1.close()


window = tk.Tk()

window.title("Awake Drive")

isPause = tk.IntVar()
pause_button = tk.Checkbutton(window, text="Dừng", variable=isPause, onvalue=1, offvalue=0)
pause_button.pack(pady=10)

start_button = tk.Button(window, text="Bắt đầu", command=start_test)
start_button.pack(pady=10)
# Raw wave
image_frame1 = tk.Label(window, width=1000, height=500, bg="white")
image_frame1.pack(side=tk.LEFT)

show_image()
# Tạo một khung văn bản để hiển thị trạng thái buồn ngủ/tỉnh táo
status_label = tk.Label(window, text="Trạng thái", font=("Arial", 14))
status_label.pack(pady=10)
state_label = tk.Label(window, text="", font=("Arial", 20))
state_label.pack(pady=10)

window.mainloop()
# Close the serial port
print("DONE")
# s.close()
# file.close()
