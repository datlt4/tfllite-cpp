import pudb
import tensorflow as tf
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from uuid import uuid4

CROP_FRAC = 0.85


get_corner_interpreter = tf.lite.Interpreter(model_path="TrainedModel/getCorners.tflite")
# get_corner_interpreter = tf.lite.Interpreter(model_path="TrainedModel/getCorners_int8.tflite")
# get_corner_interpreter = tf.lite.Interpreter(model_path="TrainedModel/getCorners_fp16.tflite")
get_corner_interpreter.allocate_tensors()
get_corner_input_details = get_corner_interpreter.get_input_details()
get_corner_output_details = get_corner_interpreter.get_output_details()
get_corner_input_shape = get_corner_input_details[0]['shape']
get_corner_input_index = get_corner_interpreter.get_input_details()[0]["index"]
get_corner_output_index = get_corner_interpreter.get_output_details()[
    0]["index"]

corner_refiner_interpreter = tf.lite.Interpreter(model_path="TrainedModel/cornerRefiner.tflite")
# corner_refiner_interpreter = tf.lite.Interpreter(model_path="TrainedModel/cornerRefiner_int8.tflite")
# corner_refiner_interpreter = tf.lite.Interpreter(model_path="TrainedModel/cornerRefiner_fp16.tflite")
corner_refiner_interpreter.allocate_tensors()
corner_refiner_input_details = corner_refiner_interpreter.get_input_details()
corner_refiner_output_details = corner_refiner_interpreter.get_output_details()
corner_refiner_input_shape = corner_refiner_input_details[0]['shape']
corner_refiner_input_index = corner_refiner_interpreter.get_input_details()[
    0]["index"]
corner_refiner_output_index = corner_refiner_interpreter.get_output_details()[
    0]["index"]


if __name__ == "__main__":
    # for path in tqdm(glob("/mnt/2B59B0F32ED5FBD7/Projects/HELIGATE/Painterly/samples/A4-dataset1/*.jpg") +
    #                  glob("/mnt/2B59B0F32ED5FBD7/Projects/HELIGATE/Painterly/samples/A4-dataset2/*.jpg") +
    #                  glob("/mnt/2B59B0F32ED5FBD7/Projects/HELIGATE/Painterly/samples/ICDAR-Dataset/*.jpg")):
    for path in ["/mnt/2B59B0F32ED5FBD7/Projects/HELIGATE/Painterly/samples/A4-dataset1/D1_10.jpg"]:
        try:
            image = cv2.imread(path)
            o_img = np.copy(image)
            myImage = np.copy(o_img)
            for _ in tqdm(range(1000)):
                img_temp = cv2.resize(myImage, (32, 32))
                get_corner_input_tensor = np.array(
                    np.expand_dims(img_temp, 0), dtype=np.float32)
                get_corner_interpreter.set_tensor(
                    get_corner_input_index, get_corner_input_tensor)
                get_corner_interpreter.invoke()
                response = get_corner_interpreter.get_tensor(
                    get_corner_output_index)[0]
                x = response[[0, 2, 4, 6]]
                y = response[[1, 3, 5, 7]]
                x = x * myImage.shape[1]
                y = y * myImage.shape[0]
                tl = myImage[max(0, int(2*y[0] - (y[3]+y[0])/2)):int((y[3]+y[0])/2),
                            max(0, int(2*x[0] - (x[1]+x[0])/2)):int((x[1]+x[0])/2)]
                tr = myImage[max(0, int(2*y[1] - (y[1]+y[2])/2)):int((y[1]+y[2])/2),
                            int((x[1]+x[0])/2):min(myImage.shape[1]-1, int(x[1]+(x[1]-x[0])/2))]
                br = myImage[int((y[1]+y[2])/2):min(myImage.shape[0]-1, int(y[2]+(y[2]-y[1])/2)),
                            int((x[2]+x[3])/2):min(myImage.shape[1]-1, int(x[2]+(x[2]-x[3])/2))]
                bl = myImage[int((y[0]+y[3])/2):min(myImage.shape[0]-1, int(y[3]+(y[3]-y[0])/2)),
                            max(0, int(2*x[3] - (x[2]+x[3])/2)):int((x[3]+x[2])/2)]
                tl = (tl, max(0, int(2*x[0] - (x[1]+x[0])/2)),
                    max(0, int(2*y[0] - (y[3]+y[0])/2)))
                tr = (tr, int((x[1]+x[0])/2), max(0, int(2*y[1] - (y[1]+y[2])/2)))
                br = (br, int((x[2]+x[3])/2), int((y[1]+y[2])/2))
                bl = (bl, max(0, int(2*x[3] - (x[2]+x[3])/2)), int((y[0]+y[3])/2))

                corner_address = []
                for b in (tl, tr, br, bl):
                    img = b[0]
                    _myImage = np.copy(img)
                    ans_x = 0.0
                    ans_y = 0.0
                    y = None
                    x_start = 0
                    y_start = 0
                    up_scale_factor = (img.shape[1], img.shape[0])
                    while(_myImage.shape[0] > 10 and _myImage.shape[1] > 10):
                        img_temp = cv2.resize(_myImage, (32, 32))
                        corner_refiner_input_tensor = np.array(
                            np.expand_dims(img_temp, 0), dtype=np.float32)
                        corner_refiner_interpreter.set_tensor(
                            corner_refiner_input_index, corner_refiner_input_tensor)
                        corner_refiner_interpreter.invoke()
                        response = corner_refiner_interpreter.get_tensor(
                            corner_refiner_output_index)
                        response_up = response[0]
                        response_up = response_up * up_scale_factor
                        y = response_up + (x_start, y_start)
                        x_loc = int(y[0])
                        y_loc = int(y[1])
                        if x_loc > _myImage.shape[1] / 2:
                            start_x = min(x_loc + int(round(_myImage.shape[1] * CROP_FRAC / 2)), _myImage.shape[1]) - int(round(_myImage.shape[1] * CROP_FRAC))
                        else:
                            start_x = max(x_loc - int(_myImage.shape[1] * CROP_FRAC / 2), 0)
                        if y_loc > _myImage.shape[0] / 2:
                            start_y = min(y_loc + int(_myImage.shape[0] * CROP_FRAC / 2), _myImage.shape[0]) - int(_myImage.shape[0] * CROP_FRAC)
                        else:
                            start_y = max(y_loc - int(_myImage.shape[0] * CROP_FRAC / 2), 0)
                        ans_x += start_x
                        ans_y += start_y
                        _myImage = _myImage[start_y:start_y + int(_myImage.shape[0] * CROP_FRAC),
                                            start_x:start_x + int(_myImage.shape[1] * CROP_FRAC)]
                        img = img[start_y:start_y + int(img.shape[0] * CROP_FRAC),
                                start_x:start_x + int(img.shape[1] * CROP_FRAC)]
                        up_scale_factor = (img.shape[1], img.shape[0])

                    ans_x += y[0]
                    ans_y += y[1]
                    temp = np.array((int(round(ans_x)), int(round(ans_y))))
                    temp[0] += b[1]
                    temp[1] += b[2]
                    corner_address.append(temp)
            for a in range(4):
                cv2.line(image, tuple(corner_address[a % 4]), tuple(
                    corner_address[(a+1) % 4]), (255, 0, 0), 2)
            cv2.imwrite(f"output4/{uuid4().hex}.jpg", image)
        except Exception as e:
            print(e, path)
