import cv2
import cv2 as cv
import numpy as np

#inicialização de parametros
confThreshold = 0.1
nmsThreshold = 0.1
inpWidth = 316
inpHeight = 316


#nomes das classes
classesFile = "obj.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

#configuração e modelo gerado
modelConfiguration = "yolov3-tiny.cfg"
modelWeights = "yolov3-tiny.backup"

#carregamento
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Desenhar boundbox
def drawPred(classId, conf, left, top, right, bottom):

    cv.rectangle(img, (left, top), (right, bottom), (255, 178, 50), 3)

    label = '%.2f' % conf


    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)


    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(img, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                 (255, 255, 255), cv.FILLED)
    cv.putText(img, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)


# Remove boundbox com confiança baixa
def postprocess(frame, outs):

    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)


# Processo
winName = 'Deep learning object detection in OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

stream = cv2.VideoCapture(0)

index = 0
while stream.isOpened():

    r, img = stream.read()

    if r ==  True:

        blob = cv.dnn.blobFromImage(img, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)


        net.setInput(blob)

        outs = net.forward(getOutputsNames(net))

        postprocess(img, outs)


        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
        cv.putText(img, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        cv2.imwrite('pep/' + index.__str__() + '.jpg', img)


        cv.imshow(winName, img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    else:
        break

cv2.destroyAllWindows()