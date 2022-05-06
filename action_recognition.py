import cv2
import imutils
import numpy as np
from openvino.inference_engine import IECore
from imutils import paths
from queue import Queue

TEST_PATH = "testImgs"
CHINA = False
CONF = 0.4

pColor = (0, 0, 255)  # plate bounding-rect and information color
rectThinkness = 2

encoder_model_bin = "./models/driver-action-recognition-adas-0002-encoder.bin"
encoder_model_xml = "./models/driver-action-recognition-adas-0002-encoder.xml"
decoder_model_bin = "./models/driver-action-recognition-adas-0002-decoder.bin"
decoder_model_xml = "./models/driver-action-recognition-adas-0002-decoder.xml"
VIDEO_PATH = "./video/driver_actions_2.mp4"
full_name_labels = "./driver_actions.txt"
encoder_results_queue = Queue(16)

device = "CPU"
actions_list = [
    "Safe driving",
    "Texting left",
    "Texting right",
    "Talking phone left",
    "Talking phone right",
    "Operating radio",
    "Drinking eating",
    "Reaching behind",
    "Hair and makeup",
]


def drawText(frame, scale, rectX, rectY, rectColor, text):

    textSize, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)

    top = max(rectY - rectThinkness, textSize[0])

    cv2.putText(
        frame, text, (rectX, top), cv2.FONT_HERSHEY_SIMPLEX, scale, rectColor, 3
    )


def center_crop(frame, crop_size):
    img_h, img_w, _ = frame.shape

    x0 = int(round((img_w - crop_size[0]) / 2.0))
    y0 = int(round((img_h - crop_size[1]) / 2.0))
    x1 = x0 + crop_size[0]
    y1 = y0 + crop_size[1]

    return frame[y0:y1, x0:x1, ...]


def adaptive_resize(frame, dst_size):
    h, w, _ = frame.shape
    scale = dst_size / min(h, w)
    ow, oh = int(w * scale), int(h * scale)

    if ow == w and oh == h:
        return frame
    return cv2.resize(frame, (ow, oh))


def preprocess_frame(frame):
    frame = adaptive_resize(frame, 224)
    frame = center_crop(frame, (224, 224))

    frame = frame.transpose((2, 0, 1))  # HWC -> CHW
    frame = frame[np.newaxis, ...]  # add batch dimension
    return frame


def actionRecognition(
    frame,
    encoder_execution_net,
    encoder_input_blob,
    decoder_execution_net,
    decoder_input_blob,
):

    encoder_blob = preprocess_frame(frame)
    # encoder_blob = cv2.dnn.blobFromImage(frame, size=(224, 224), ddepth=cv2.CV_8U)
    encoder_result = encoder_execution_net.infer(
        inputs={encoder_input_blob: encoder_blob}
    ).get("513")
    # print(encoder_results["513"])
    encoder_results_queue.put(encoder_result)
    index = 0

    if encoder_results_queue.full():
        decoder_input = np.array(list(encoder_results_queue.queue)).reshape(1, 16, 512)
        decoder_results = (
            decoder_execution_net.infer(inputs={decoder_input_blob: decoder_input})
            .get("804")[0]
            .tolist()
        )
        index = decoder_results.index(max(decoder_results))
        encoder_results_queue.get()

    print(actions_list[index])
    drawText(frame, 600 * 0.008, 10, 10, pColor, actions_list[index])
    showImg = imutils.resize(frame, height=600)
    cv2.imshow("showImg", showImg)


def main():

    ie = IECore()

    encoder_neural_net = ie.read_network(
        model=encoder_model_xml, weights=encoder_model_bin
    )
    encoder_input_blob = next(iter(encoder_neural_net.input_info))
    encoder_neural_net.batch_size = 1
    encoder_execution_net = ie.load_network(
        network=encoder_neural_net, device_name=device.upper()
    )

    decoder_neural_net = ie.read_network(
        model=decoder_model_xml, weights=decoder_model_bin
    )
    decoder_input_blob = next(iter(decoder_neural_net.input_info))
    decoder_neural_net.batch_size = 1
    decoder_execution_net = ie.load_network(
        network=decoder_neural_net, device_name=device.upper()
    )

    vidcap = cv2.VideoCapture(VIDEO_PATH)
    success, img = vidcap.read()
    while success:
        actionRecognition(
            img,
            encoder_execution_net,
            encoder_input_blob,
            decoder_execution_net,
            decoder_input_blob,
        )
        cv2.waitKey(1)
        if cv2.waitKey(10) == 27:  # exit if Escape is hit
            break
        success, img = vidcap.read()
    print("chau")


if __name__ == "__main__":
    main()
