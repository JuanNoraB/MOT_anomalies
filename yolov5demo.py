import argparse
import numpy 
import torch
import yolov5
import pandas
from typing import Union, List, Optional
#
import cv2
import norfair
from norfair import Detection, Tracker, Video, Paths

max_distance_between_points: int = 30
df = pandas.DataFrame(columns=['x','y','width','height','score','category','id'])

class YOLO:
    def __init__(self, model_path: str, device: Optional[str] = None):
        self.model = yolov5.load(model_path, device="cpu")

    def __call__(
        self,
        img: Union[str, numpy.ndarray], #A Union is a type that can hold any of the types that are defined within it. In this case, it can hold any type.
        conf_threshold: float = 0.25, #Confidence threshold is a parameter in a machine learning model that sets the minimum threshold for predictions.
        iou_threshold: float = 0.45, #The IOU threshold is the minimum threshold for the Intersection over Union score that is required for two bounding boxes to be considered a match.
        image_size: int = 720,
        classes: Optional[List[int]] = None
    ) -> torch.tensor:

        self.model.conf = conf_threshold
        self.model.iou = iou_threshold
        if classes is not None:
            self.model.classes = classes
        detections = self.model(img, size=image_size)
        return detections


def euclidean_distance(detection, tracked_object):
    #print(detection.points, tracked_object)
    return numpy.linalg.norm(detection.points - tracked_object.estimate)


def center(points):
    #print(points)
    return [numpy.mean(numpy.array(points), axis=0)]


def yolo_detections_to_norfair_detections(
    yolo_detections: torch.tensor,
    track_points: str = 'centroid'  # bbox or centroid
) -> list: #List[Detection]:
    """convert detections_as_xywh to norfair detections
    """
    norfair_detections: List[Detection] = []
    xywh_target = []
    if track_points == 'centroid':

        detections_as_xywh = yolo_detections.xywh[0]
        #detections_as_xywh = yolo_detections.pred[0][:, :6]
        #delect all elements in detections_as_xywh that has 0.000 as last element
        #make a copy of a tensor
        detections_as_xywh1 = detections_as_xywh.clone() 
        #detections_as_xywh1 = torch.tensor([i for i in detections_as_xywh1 if int(i[2]) < 400.00])
        detections_as_xywh1 =detections_as_xywh1.numpy()
        detections_as_xywh1 = numpy.array([i for i in detections_as_xywh1 if int(i[-1]) < 1.00])
        detections_as_xywh1 = numpy.array([i for i in detections_as_xywh1 if float(i[-2]) > 0.50])
        detections_as_xywh = torch.from_numpy(detections_as_xywh1)
        #for i,j in enumerate(detections_as_xywh1):
         #   if int(j[-1]) < 1:
          #      print(detections_as_xywh1[i])
                
        #convert array in a teosor
        print(detections_as_xywh1)
        for detection_as_xywh in detections_as_xywh:
            
            
            centroid = numpy.array(
                [
                    detection_as_xywh[0].item(),
                    detection_as_xywh[1].item()
                ]
            )
            scores = numpy.array([detection_as_xywh[4].item()])
            norfair_detections.append(
                Detection(points=centroid, scores=scores)
            )
            xywh_target.append((detection_as_xywh,norfair_detections[-1]))
           
            
    elif track_points == 'bbox':
        print("bbox")
        detections_as_xyxy = yolo_detections.xyxy[0]
        for detection_as_xyxy in detections_as_xyxy:
            bbox = numpy.array(
                [
                    [detection_as_xyxy[0].item(), detection_as_xyxy[1].item()],
                    [detection_as_xyxy[2].item(), detection_as_xyxy[3].item()]
                ]
            )
            scores = numpy.array([detection_as_xyxy[4].item(), detection_as_xyxy[4].item()])
            norfair_detections.append(
                Detection(points=bbox, scores=scores)
            )
    #print(norfair_detections)
    
    #print('\n')
    #print(xywh_target)
    #print("-----------------------------------------------------")
    #print(norfair_detections)
    #print([i[1] for i in xywh_target] == norfair_detections)
    #print('\n')
    return xywh_target #norfair_detections


parser = argparse.ArgumentParser(description="Track objects in a video.")
parser.add_argument("files", type=str, nargs="+", help="Video files to process")
parser.add_argument("--detector_path", type=str, default="yolov5m6.pt", help="YOLOv5 model path")
parser.add_argument("--img_size", type=int, default="720", help="YOLOv5 inference size (pixels)")
parser.add_argument("--conf_thres", type=float, default="0.25", help="YOLOv5 object confidence threshold")
parser.add_argument("--iou_thresh", type=float, default="0.45", help="YOLOv5 IOU threshold for NMS")
parser.add_argument('--classes', nargs='+', type=int, help='Filter by class: --classes 0, or --classes 0 2 3')
parser.add_argument("--device", type=str, default=None, help="Inference device: 'cpu' or 'cuda'")
parser.add_argument("--track_points", type=str, default="centroid", help="Track points: 'centroid' or 'bbox'")
args = parser.parse_args()
 
contador=0
model = YOLO(args.detector_path, device=args.device)

for input_path in args.files:
    video = Video(input_path=input_path)
    tracker = Tracker(
        distance_function=euclidean_distance,
        distance_threshold=max_distance_between_points,
    )
    paths_drawer = Paths(center, attenuation=0.01)
    #print("Center is : ",center)
    for frame in video:
        #print(frame.shape)
        #cv2.imshow('imagen numero %d'%contador, frame)
        contador+=1
        yolo_detections = model(
            frame,
            conf_threshold=args.conf_thres,
            iou_threshold=args.iou_thresh,
            image_size=args.img_size,
            classes=args.classes
        )
        #predictions= yolo_detections.pred[0]
        #print(predictions)
        #boxesp = predictions[:, :4] # x1, y1, x2, y2
        #scoresp = predictions[:, 4]
        #categoriesp = predictions[:, 5]
        #print("predictions",predictions)
        #print("boxes",boxesp)
        #print("scores",scoresp)
        #print("categories",categoriesp)
        #print("tipo de args.file : ",type(args.files),"\n")
        #print(args.files,'\n')
        #print(args)
        detections = yolo_detections_to_norfair_detections(yolo_detections, track_points=args.track_points)
        detectionsdf= detections
        
        #change the secont value of the tuple to the detection by tracker.update([i[1] for i in detections)
        #detections1 = [(i[0],tracker.update(detections=i[1])) for i in detections]
        #print(detections1)
        print("--------")
        #print(detectionsdf)
        try:
            for i in detectionsdf:
                #first row of df with all 0 
                df.loc[contador-1]=[float(i[0][0]),float(i[0][1]),float(i[0][2]),float(i[0][3]),float(i[0][4]),float(i[0][5]),str(i[1])]
                #print(float(i[0][0]),float(i[0][1]),float(i[0][2]),float(i[0][3]),float(i[0][4]),float(i[0][5]),str(i[1]))
                #print(i[0][0],i[0][1],i[0][2],i[0][3],i[0][4],i[0][5],i[1])
        except:
            print("Error")
        detections = [i[1] for i in detections]
        tracked_objects = tracker.update(detections=detections)
        print("*********************************que")
        #print(getattr(tracked_objects,'count'))
        #print([1,2,3,].__class_getitem__(1))       
        try:
            print(tracked_objects.initializing_id)
            print(tracked_objects)
            print("-*****************************************-")
            
        except:
            print("Error")
        
        #if len(tracked_objects) > 0:
            #print(dir(tracked_objects[0]))
        #print(len(tracked_objects))
        #print(dir(tracked_objects))
        if args.track_points == 'centroid':
            norfair.draw_points(frame, detections)
        elif args.track_points == 'bbox':
            norfair.draw_boxes(frame, detections)
        norfair.draw_tracked_objects(frame, tracked_objects)
        frame = paths_drawer.draw(frame, tracked_objects)
        #print("tipydetection",type(detections))
        #print("*****",detections)
        #print("tacked_objects",tracked_objects)
        video.write(frame)

        #create a data frame empty with the same 8 columns
        
       
        
#print(args.files[0])
#print(contador)
print(df)

#save dataframe df
name = args.files[0]
name = name[name.rfind('/')+1:name.rfind('.')]
df.to_csv(f'{name}.csv',index=False)
