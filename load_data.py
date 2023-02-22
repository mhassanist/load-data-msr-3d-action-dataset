import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import cv2
import struct
import os

# Define the path to the dataset directory
dataset_path = "Depth/"
action_label_dict = {
    "a01": "High Arm Wave",
    "a02": "Horizontal Arm Wave",
    "a03": "Hammer",
    "a04": "Hand Catch",
    "a05": "Forward Punch",
    "a06": "High Throw",
    "a07": "Draw X",
    "a08": "Draw Tick",
    "a09": "Draw Circle",
    "a10": "Hand Clap",
    "a11": "Two Hand Wave",
    "a12": "Side-Boxing",
    "a13": "Bend",
    "a14": "Forward Kick",
    "a15": "Side Kick",
    "a16": "Jogging",
    "a17": "Tennis Swing",
    "a18": "Tennis Serve",
    "a19": "Golf Swing",
    "a20": "Pick Up and Throw"
}

action_labels = {}

for filename in os.listdir(dataset_path):
    if not filename.endswith("depth.bin"):
        continue

    action_label = filename.split("_")[0]

    print(action_label)
    if action_label not in action_labels:
        action_labels[action_label] = action_label_dict[action_label]
    else:
        continue

    # Load the depth map data from the binary file
    with open(dataset_path + filename, "rb") as f:
        i,  = struct.unpack('i', f.read(4))  # frame count
        w,  = struct.unpack('i', f.read(4))  # width
        h,  = struct.unpack('i', f.read(4))  # height

        print(i, w, h)

        depth_data = np.fromfile(f, dtype=np.uint32)
        depth_map = depth_data.reshape((i, h, w))

        # You can show one single image by uncommenting the following lines
        # plt.imshow(depth_map[32], cmap='gray')
        # plt.show()
        # break

        output_video_path = "video_action{}.avi".format(action_label)

        # specify the frame rate of the output video
        fps = 15

        # create a VideoWriter object to write the output video
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(
            output_video_path, fourcc, fps, (w, h), isColor=False)

        # iterate over each depth frame and write it to the output video
        for i in range(i):
            # normalize the depth map to 8-bit grayscale values for visualization
            depth_frame = (
                depth_map[i]/np.max(depth_map[i])*255).astype(np.uint8)
            # write the current depth frame to the output video
            video_writer.write(depth_frame)

        # release the VideoWriter object and close the output video file
        video_writer.release()
print(action_labels)
