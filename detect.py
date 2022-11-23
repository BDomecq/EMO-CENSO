#!/usr/bin/env python
import sys
import pandas as pd
from sort import *
import logging

#from termcolor import colored
#from sort import *
#import sys
#import argparse
from utils.datasets import *
from utils.utils import *
from utils.parser import get_config
from deep_sort import build_tracker
import cv2
import torch
#import torchvision.transforms as transforms
#from matplotlib.pyplot import imshow

py_dll_path = os.path.join(sys.exec_prefix, 'Library', 'bin')
os.environ['PATH'] += py_dll_path

"""
python detect.py --weights ./weights/yolov5x.pt --img 416 --source .\inference\images\short_video_pedestrian.mp4 --device 0 --deepsort .\config\deep_sort.yaml

python detect.py --weights ./weights/yolov5x.pt --img 416 --source .\inference\images\1.mp4 --device 0 --deepsort .\config\deep_sort.yaml
"""

""" 
__author__ = ["Facundo Mercado, Brian Domecq"]
__base_implementation__ = "https://github.com/lanmengyiyu/yolov5-deepmar"
__contact__ = "facundomercado@deutschebahn.com"
__copyright__ = "Copyright 2022, DB Engineering & Consulting Gmbh"
__date__ = "2022/08/27"
__deprecated__ = False
__maintainer__ = "developer"
__status__ = "development"
__version__ = "0.0.1"
__Project__ = "Emova Censo"
__Python__Version__ = [3.10]
__venv__ = venv-mp
"""


logging.basicConfig(filename='/tmp/execution.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger=logging.getLogger(__name__)

class TrackableObject:
    """
    Defining a person that can be tracked.
    """

    def __init__(self, objectID, trajectory):
        """
        :parameter objectID: id of the object being tracked.
        :parameter trajectory: list of centroids for the object being tracked.
        :return object.
        """
        self.objectID = objectID
        self.trajectory = trajectory
        #Multiple zonas
        self.counted_1 = False
        self.counted_2 = False
        self.counted_3 = False


class PolylineDrawing:
    """
    Interactive Polyline drawer.
    """
    coordinates = []
    margin_upper = []
    margin_lower = []
    GREEN = (0, 255, 0)

    def __init__(self, img, window_name, axis, margin):
        """
        Constructor
        :parameter img: image to draw upon.
        :parameter window_name: self-explanatory.
        :parameter axis: closest axis.
        :parameter margin: detection margin.
        """
        self.img = img
        self.delta = margin

        cv2.namedWindow(winname=window_name)
        cv2.setMouseCallback(window_name, self.mouse, axis)
        while True:
            cv2.imshow(window_name, img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def mouse(self, event, x, y, flags, param):
        """
        :parameter event: mouse event.
        :parameter x: x-coordinate.
        :parameter y: y-coordinate.
        :parameter flags: callback flags.
        :parameter param: additional kwargs.
        :parameter axis: approximate axis.
        Mouse callback for left button down events.
        :return:
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            if not param:
                self.margin_upper.append((x, y + self.delta))
                self.margin_lower.append((x, y - self.delta))
            else:
                self.margin_upper.append((x + self.delta, y))
                self.margin_lower.append((x - self.delta, y))
            self.coordinates.append((x, y))

            
            cv2.polylines(self.img, np.array([self.coordinates]), isClosed=False,
                          color=(0xFF, 0xD7, 0), thickness=1)
            cv2.circle(self.img, (x, y), 4, (0xFF, 0xD7, 0), -1)
            cv2.putText(self.img, "({} , {})".format(x, y), (x + 10, y + 5), 0, 0.3, (0xFF, 0xD7, 0))


def detect(source, model, deepsort, output, img_size, threshold, margin,
           iou_threshold, flip_roi_axis, show_axis, fourcc, device,
           trajectory, view_frame, save_source, agnostic_nms, augment):
    """
    Main Detector & Tracker
    :parameter source: Input video.
    :parameter model: NN Model weights.
    :parameter deepsort: Enable deepsort tracker.
    :parameter output: Inference output path.
    :parameter img_size: Input image size.
    :parameter threshold: NN detection threshold.
    :parameter margin: detection margin.
    :parameter iou_threshold: IOU overlap.
    :parameter roi_position: ROI line position.
    :parameter flip_roi_axis: Whether to flip ROI axis (x to y).
    :parameter show_axis: Show axis line.
    :parameter fourcc: Concerning output video format.
    :parameter device: CUDA or cpu.
    :parameter trajectory: Whether to display trajectory.
    :parameter view_frame: Whether to display real-time detection.
    :parameter save_source: Whether to save the image.
    :parameter save_results: Whether to save results.
    :parameter agnostic_nms: Agnostic NMS flag.
    :parameter augment: augment flag.
    :return: void.
    """
     #Predifining flow dict
    flow_dict = {'Execution Time [s]': None,
                'Passenger Inflow': [],
                'Passenger Outflow': []}

    out, source, weights, imgsz, trajectory_flag = \
        output, source, model, img_size, trajectory

    device = torch_utils.select_device(device)

    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    half = device.type != device #'cpu'

    if deepsort:
        mot_tracker = build_tracker(cfg, use_cuda=True)  # Swap for CUDA
    else:
        mot_tracker = Sort()
    trajectory = {}

    model = torch.load(weights, map_location=device)['model'].float()
    model.to(device).eval()

    if half:
        model.half()

    vid_path, vid_writer = None, None

    # Loading frames
    dataset = LoadImages(source, img_size=imgsz)

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(32)]

    #------------------------------------------------------
    # Detection Zones --> PASAR A PARAMETRO
    zone = int(input('Cantidad de zonas a seleccionar: '))
    
    #------------------------------------------------------
    # Drawing initial ROI\s
    roi_coordinates = []
    roi_margin_lower = []
    roi_margin_upper = []

    for x in range(zone):
        for path, img, im0s, vid_cap in dataset:
            roi = PolylineDrawing(im0s, 'first', flip_roi_axis, margin)
            break
    roi_coordinates.append(roi.coordinates)
    roi_margin_lower.append(roi.margin_lower)
    roi_margin_upper.append(roi.margin_upper)
  
    ###########################################################################
    #ZONA PREDEFINIDA
        
    #Calculo de las rectas de las zonas
    m_l = []
    b_l = []
    m_u = []
    b_u = []
    
    for x in roi_margin_lower:
        roi_lower = list(x)
    for x in roi_margin_upper:
        roi_upper = list(x)
    for x in roi_coordinates:
        roi_coor = list(x)

    for x in range(zone):
        #ZONE 1
        m_l.append((roi_lower[x*2][1]-roi_lower[x*2+1][1])/(roi_lower[x*2][0]-roi_lower[x*2+1][0]))
        m_u.append((roi_upper[x*2][1]-roi_upper[x*2+1][1])/(roi_upper[x*2][0]-roi_upper[x*2+1][0]))
        b_l.append(-1*m_l[0]*roi_lower[x*2][0] + roi_lower[x*2][1])
        b_u.append(-1*m_l[0]*roi_upper[x*2][0] + roi_upper[x*2][1])
    
    ###########################################################################

    # Trackable object collection
    trackableObjects = {}
    counter = [0, 0, 0, 0, 0, 0]
    a_u = [0, 0, 0] #line upper
    a_l = [0, 0, 0] #line lower
    #--------------
    #Agregado CSV
    #--------------
    #count = dict()
    
    #--------------

    # Start Inference
    t0 = time.time() #processing time
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)
    _ = model(img.half() if half else img) if device.type != device else None

    # Work on every frame
    for path, img, im0s, vid_cap in dataset:

        try:
            fps = vid_cap.get(cv2.CAP_PROP_FPS)
            frames_to_push = 1*60*fps
            print("frames push: ", frames_to_push)

            if frames_to_push > dataset.nframes:
                print("Cannot reach the 15 minute push interval.")

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Execute inference
            t1 = torch_utils.time_synchronized()
            prediction = model(img, augment=augment)[0]

            # Apply NMS
            prediction = non_max_suppression(prediction, threshold, opt.iou_thr, agnostic=agnostic_nms)
            t2 = torch_utils.time_synchronized()

            # Process detections: detections per image.
            for i, det in enumerate(prediction):
                p, s, im0 = path, '', im0s

                save_path = str(Path(out) / Path(p).name) #ESTO DEBERIA VOLAR
                s += '%gx%g ' % img.shape[2:]
                if det is not None and len(det):
                    # Rescale bounding boxes.
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])

                    if opt.deepsort:
                        track_det = [elem[:4].tolist() for elem in det if elem[5] == 0]
                        track_det_xywh = xyxy2xywh(np.array(track_det))
                        cls_conf = [elem[4].tolist() for elem in det if elem[5] == 0]
                        track_bbs_ids = mot_tracker.update(track_det_xywh, cls_conf, im0)
                    else:
                        track_det = [elem[:5].tolist() for elem in det if elem[5] == 0]
                        track_det = np.array(track_det)
                        track_bbs_ids = mot_tracker.update(track_det)

                    # Normalized and non-normalized bounding box coordinates.
                    for *xyxy, pid in track_bbs_ids:
                        # Annotate frames with their bounding boxes
                        if save_source or view_frame:
                            label = '%d' % pid

                            if label not in trajectory:
                                trajectory[label] = []
                            plot_one_box(xyxy, im0, label=label, color=colors[int(label) % 32], line_thickness=1)
                            height, width, _ = im0.shape
                            x1, y1, x2, y2 = max(0, int(xyxy[0])), max(0, int(xyxy[1])), \
                                            min(width, int(xyxy[2])), min(height, int(xyxy[3]))
                            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                            trajectory[label].append(center)


                            # ROI Trespassing logic
                            to = trackableObjects.get(label, None)                        
                            if to is None:
                                to = TrackableObject(label, trajectory[label])    
                            else:
                                for x in range(zone):
                                    a_u[x] = (int(m_u[x]*center[0] + b_u[x]))
                                    a_l[x] = (int(m_l[x]*center[0] + b_l[x]))

                                y = [c[1] for c in to.trajectory]
                                direction = center[1] - np.mean(y)

                                if center[1] > a_u[0] and np.mean(y) < a_u[0] and direction > 0 and not to.counted_1:
                                                                        
                                    if center[1] > a_l[0] and ( center[0] >= roi_coor[0][0] and center[0] <= roi_coor[1][0]): #termine de pasar
                                        counter[1] +=1 #Out
                                        to.counted_1 = True

                                elif center[1] < a_l[0] and np.mean(y) > a_l[0] and direction < 0 and not to.counted_1:
                                                
                                    #Agregado conteo
                                    if center[1] < a_u[0] and (center[0] >= roi_coor[0][0] and center[0] <= roi_coor[1][0]): #termine de pasar
                                        counter[0] +=1
                                        to.counted_1 = True
                                
                                if zone == 2 or zone == 3 :
                                    #Zone 2
                                    if center[1] > a_u[1] and np.mean(y) < a_u[1] and direction > 0 and not to.counted_2:
                                        if center[1] > a_l[1] and ( center[0] >= roi_coor[2][0] and center[0] <= roi_coor[3][0]): #termine de pasar
                                            counter[3] +=1 #Out
                                            to.counted_2 = True
                                                
                                elif center[1] < a_l[1] and np.mean(y) > a_l[1] and direction < 0 and not to.counted_2:
                                    #Agregado conteo
                                    if center[1] < a_u[1]and ( center[0] >= roi_coor[2][0] and center[0] <= roi_coor[3][0]): #termine de pasar
                                        counter[2] +=1
                                        to.counted_2 = True
                                if zone == 3:
                                    #Zone 3
                                    if center[1] > a_u[2] and np.mean(y) < a_u[2] and direction > 0 and not to.counted_3:
                                        if center[1] > a_l[2] and ( center[0] >= roi_coor[4][0] and center[0] <= roi_coor[5][0]): #termine de pasar
                                            counter[5] +=1 #Out
                                            to.counted_3 = True
                                                
                                elif center[1] < a_l[2] and np.mean(y) > a_l[2] and direction < 0 and not to.counted_3:
                                    #Agregado conteo
                                    if center[1] < a_u[2]and ( center[0] >= roi_coor[4][0] and center[0] <= roi_coor[5][0]): #termine de pasar
                                        counter[4] +=1
                                        to.counted_3 = True
                                                        
                            to.trajectory.append(center)
                            trackableObjects[label] = to

                            # Drawing trajectory
                            for i in range(1, len(trajectory[label])):
                                if trajectory[label][i - 1] is None or trajectory[label][i] is None:
                                    continue
                                if trajectory_flag:
                                    cv2.arrowedLine(im0, trajectory[label][i - 1], trajectory[label][i],
                                                    colors[int(label) % 32], 1, tipLength=0.8)
                                    cv2.circle(im0, (center[0], center[1]), 2, colors[int(label) % 32], -1)

                            # Drawing the ROI.
                            if show_axis:
                                #Zona 1
                                cv2.line(im0, roi_margin_lower[0][0], roi_margin_lower[0][1] ,
                                    color=(0, 0, 255), thickness=1)
                                cv2.line(im0, roi_margin_upper[0][0], roi_margin_upper[0][1] ,
                                    color=(0, 0, 255), thickness=1)
                                #Zona 2                            
                                if zone == 2 or zone ==3:    
                                    cv2.line(im0, roi_margin_lower[0][2], roi_margin_lower[0][3] ,
                                            color=(255, 0, 0), thickness=1)
                                    cv2.line(im0, roi_margin_upper[0][2], roi_margin_upper[0][3] ,
                                            color=(255, 0, 0), thickness=1)
                                if zone == 3: 
                                    cv2.line(im0, roi_margin_lower[0][4], roi_margin_lower[0][5] ,
                                            color=(0, 255, 0), thickness=1)
                                    cv2.line(im0, roi_margin_upper[0][4], roi_margin_upper[0][5] ,
                                            color=(0, 255, 0), thickness=1)
                           

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    #Zona 1
                    cv2.putText(im0, f'(out): {counter[1]} | (in): {counter[0]}', (
                            10, 30), font, 0.5, (0, 0, 255), 2, cv2.FONT_HERSHEY_SIMPLEX)
                    #Zona 2
                    if zone == 2 or zone == 3:
                        cv2.putText(im0, f'(out): {counter[3]} | (in): {counter[2]}', (
                            10, 50), font, 0.5, (255, 0, 0), 2, cv2.FONT_HERSHEY_SIMPLEX)
                    #Zona 3
                    if zone == 3:
                        cv2.putText(im0, f'(out): {counter[5]} | (in): {counter[4]}', (
                            10, 70), font, 0.5, (0, 255, 0), 2, cv2.FONT_HERSHEY_SIMPLEX)

                # Store partial results
                if dataset.frame % frames_to_push == 0:
                    inflow = counter[2]
                    outflow = counter[3]
                    flow_dict['Passenger Inflow'].append(inflow)
                    flow_dict['Passenger Outflow'].append(outflow)
                    


                # Showcase results
                if not view_frame:
                    cv2.imshow(p, im0)
                    if cv2.waitKey(1) == ord('q'):
                        raise StopIteration

                # Store image with detections
                if save_source:
                    if dataset.mode == 'images':
                        cv2.imwrite(save_path, im0)
                    else:
                        if vid_path != save_path:
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()

                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                        vid_writer.write(im0)
        except Exception as e:
            logger.error(e)
            continue
    
    execution_time = time.time() - t0
    flow_dict['Execution Time [s]'] = str(execution_time)

    # if save_results:
    # TODO despues de montar la logica de direccion, emitir un reporte con personas.

    #Tiempo de ejecucion
    print('Done. (%.3fs)' % (time.time() - t0))
    #Print coordenadas por zonas
    print("Coordenadas: ", roi_coordinates)
    print('zone 1: ' + f'(out): {counter[1]} | (in): {counter[0]}')
    print('zone 2: ' + f'(out): {counter[3]} | (in): {counter[2]}')
    print('zone 3: ' + f'(out): {counter[5]} | (in): {counter[4]}')

    #SAVE csv
    #exp_results(count)

    point_dict = {'ID': '',
              'Line': '',
              'Station': '',
              'Windpass': '',
              'Direction': '',
              'Is Access': ''}

    run_dict = {'Source': source,
            'Agnostic NMS': agnostic_nms,
            'Detection Threshold': threshold,
            'IOU Overlap': iou_threshold,
            'roi_coordinates': roi_coordinates}
    
    point_df = pd.DataFrame(point_dict)
    flow_df = pd.DataFrame(flow_dict)
    run_df = pd.DataFrame(run_dict)
    results_df = pd.concat([point_df, flow_df, run_df], axis="columns")
    results_df.to_csv(os.path.join(os.getcwd(), 'result.csv'))

#SALIDA CSV    
def exp_results(count):
    f = open ("resultado.csv", "w")
    #f.write('Personas;Sentido;Time;Source;\n')
    f.write('Personas;Sentido;Source;\n')

        
    for x in count.keys():
        sentido = str(count[x][0]).replace('.',',')
        source = str(count[x][1]).replace('.',',')
        #process_time = str(count[x][2]).replace('.',',')
        
        f.write( str(x) + ";" + sentido + ';' + source + ";" + "\n" )
        #f.write( str(x) + ";" + cont_in + ';' + cont_out + ";" + process_time + "\n" )
            
    f.close ()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Object tracker and detector.')

    parser.add_argument('-s', '--source', type=str,
                        default='inference/images', help='Source.')

    parser.add_argument('-w', '--weights', type=str,
                        default='weights/yolov5s.pt', help='Model.pt path.')

    parser.add_argument('-deep', '--deepsort', type=str,
                        default="", help='Whether to use deepsort or sort.')

    parser.add_argument('-margin', '--margin', type=int,
                        default=50, help='Detection margins.')

    parser.add_argument('-o', '--output', type=str,
                        default='inference/output', help='Output folder.')

    parser.add_argument('-i_size', '--img_size', type=int,
                        default=640, help='Inference size (pixels).')

    parser.add_argument('-thr', '--detection_thr', type=float,
                        default=0.01, help='Object confidence threshold.')

    parser.add_argument('-iou', '--iou_thr', type=float,
                        default=0.04, help='IOU threshold for NMS.')

    parser.add_argument('-sa', '--show_axis', type=bool,
                        default=True, help='Axis flag.')

    parser.add_argument('-ax', '--flip_axis', default=False, type=bool, help='ROI Axis (x by default).')

    parser.add_argument('-d', '--device', type=str,
                        default='', help='Cuda device, i.e. 0 or 0,1,2,3 or cpu.')

    parser.add_argument('-tr', '--trajectory', type=bool,
                        default=False, help='Turns trajectory on or off.')

    parser.add_argument('-fourcc', '--fourcc', type=str,
                        default='mp4v', help='Output video codec.')

    parser.add_argument('-v', '--view_frame', type=bool,
                        default=False, help='Display results.')

    parser.add_argument('-save_source', '--save_source',
                        default=True, action='store_true', help='Save source.')

    parser.add_argument('-agn', '--agnostic_nms',
                        action='store_true', help='Class-agnostic NMS.')

    parser.add_argument('-aug', '--augment',
                        action='store_true', help='augmented inference')

    opt = parser.parse_args()

    opt.img_size = check_img_size(opt.img_size)

    if opt.deepsort:
        cfg = get_config()
        cfg.merge_from_file(opt.deepsort)
    #resize = 224
    #mean = [0.485, 0.456, 0.406]
    #std = [0.229, 0.224, 0.225]

    #input_transform = transforms.Compose([
    #    transforms.Resize(resize),
    #    transforms.ToTensor(),
    #    transforms.Normalize(mean=mean, std=std),
    #])

    with torch.no_grad():
        detect(source=opt.source, model=opt.weights, deepsort=opt.deepsort,
               output=opt.output, img_size=opt.img_size, threshold=opt.detection_thr, margin=opt.margin,
               iou_threshold=opt.iou_thr, flip_roi_axis=opt.flip_axis,
               show_axis=opt.show_axis, device=opt.device, trajectory=opt.trajectory,
               fourcc=opt.fourcc, view_frame=opt.view_frame, save_source=opt.save_source,
               agnostic_nms=opt.agnostic_nms, augment=opt.augment)
