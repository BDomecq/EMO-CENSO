#!/usr/bin/env python

import sys
import pandas as pd
from sort import *

from utils.datasets import *
from utils.utils import *
from utils.parser import get_config
from deep_sort import build_tracker
import cv2
import torch

py_dll_path = os.path.join(sys.exec_prefix, 'Library', 'bin')
os.environ['PATH'] += py_dll_path

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
        self.counted = False


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
    :parameter flip_roi_axis: Whether to flip ROI axis (x to y).
    :parameter show_axis: Show axis line.
    :parameter fourcc: Concerning output video format.
    :parameter device: CUDA or cpu.
    :parameter trajectory: Whether to display trajectory.
    :parameter view_frame: Whether to display real-time detection.
    :parameter save_source: Whether to save the image.
    :parameter agnostic_nms: Agnostic NMS flag.
    :parameter augment: augment flag.
    :return: void.
    """
    out, source, weights, imgsz, trajectory_flag = \
        output, source, model, img_size, trajectory

    device = torch_utils.select_device(device)

    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    half = device.type != device

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

    dataset = LoadImages(source, img_size=imgsz)

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(32)]

    """
    # Drawing initial ROI\s
    roi_coordinates = None
    roi_margin_lower = None
    roi_margin_upper = None
    for path, img, im0s, vid_cap in dataset:
        roi = PolylineDrawing(im0s, 'first', flip_roi_axis, margin)
        roi_coordinates = roi.coordinates
        roi_margin_lower = roi.margin_lower
        roi_margin_upper = roi.margin_upper
        roi_coordinates = [(80,154),(410,271)] 
        break
    """

    ###########################################################################
    #ZONA PREDEFINIDA
    roi_coordinates = [(80, 154), (410, 271)]
    ###########################################################################

    trackableObjects = {}
    counter = [0, 0, 0, 0]

    # Start Inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)
    _ = model(img.half() if half else img) if device.type != device else None

    # Work on every frame
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Execute inference
        t1 = torch_utils.time_synchronized()
        prediction = model(img, augment=augment)[0]

        # Apply NMS
        prediction = non_max_suppression(prediction, threshold, iou_threshold, agnostic=agnostic_nms)
        torch_utils.time_synchronized()

        # Process detections: detections per image.
        for i, det in enumerate(prediction):
            p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
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
                            if flip_roi_axis and not to.counted:
                                x = [c[0] for c in to.trajectory]
                                direction = center[0] - np.mean(x)

                                if any(center[0] > coord for coord in x_circle_roi) \
                                        and any(np.mean(x) < coord for coord in x_circle_roi) and direction > 0:
                                    counter[1] += 1
                                    to.counted = True
                                elif any(center[0] < coord for coord in x_circle_roi) \
                                        and any(np.mean(x) > coord for coord in x_circle_roi) and direction < 0:
                                    counter[0] += 1
                                    to.counted = True

                            elif not flip_roi_axis and not to.counted:

                                y = [c[1] for c in to.trajectory]
                                direction = center[1] - np.mean(y)

                                x = [c[0] for c in to.trajectory]
                                direction_x = center[0] - np.mean(x)

                                if any(center[1] > coord for coord in y_circle_roi)\
                                        and any(np.mean(y) < coord for coord in y_circle_roi)\
                                        and direction > 0:

                                    if center[1] > max(y_circle_roi) and ( center[0] >= min(x_circle_roi) and center[0] <= max(x_circle_roi)):

                                    #if all(center[1] > coord for coord in y_circle_roi) and (center[0] > min(x_circle_roi) and center[0] < max(x_circle_roi)):
                                        counter[3] += 1 #Out
                                        to.counted = True

                                        #count[counter[3]]=[str('Out'),str(source)]
                                        #count[counter[3]]=[str('Out'), str(source), str(time.strftime('%H:%M:%S', time.localtime()))]

                                elif any(center[1] < coord for coord in y_circle_roi) \
                                        and any(np.mean(y) > coord for coord in y_circle_roi)\
                                        and direction < 0:
                                    #Agregado conteo
                                    #if (center[1] < coord for coord in y_circle_roi): #termine de pasar
                                    if center[1] < min(y_circle_roi) and ( center[0] >= min(x_circle_roi) and center[0] <= max(x_circle_roi)): #termine de pasar

                                        counter[2] +=1
                                        to.counted = True

                                        #count[counter[2]]=[str('In'),str(source)]
                                        #count[counter[2]]=[str('In'),str(source), str(time.strftime('%H:%M:%S', time.localtime()))]

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

                            cv2.polylines(im0, np.array([roi_coordinates]), isClosed=False,
                                          color=(0xFF, 0xD7, 0), thickness=1)
                            #cv2.polylines(im0, np.array([roi_margin_lower]), isClosed=False,
                            #              color=(0xFF, 0xD7, 0), thickness=1)
                            #cv2.polylines(im0, np.array([roi_margin_upper]), isClosed=False,
                            #              color=(0xFF, 0xD7, 0), thickness=1, lineType=8)

                            x_circle_roi = [z[0] for z in roi_coordinates]
                            y_circle_roi = [z[1] for z in roi_coordinates]
                            for (x_c, y_c) in zip(x_circle_roi, y_circle_roi):
                                cv2.circle(im0, (x_c, y_c), 4, (0xFF, 0xD7, 0), -1)

                            cv2.line(im0, (x_circle_roi[0], y_circle_roi[0]), (x_circle_roi[1], y_circle_roi[0]),color=(0xFF, 0xD7, 0), thickness=1)
                            cv2.line(im0, (x_circle_roi[0], y_circle_roi[1]), (x_circle_roi[1], y_circle_roi[1]),color=(0xFF, 0xD7, 0), thickness=1)

                font = cv2.FONT_HERSHEY_SIMPLEX
                if flip_roi_axis:
                    cv2.putText(im0, f'(out): {counter[0]} | (in): {counter[1]}', (
                        10, 50), font, 0.5, (0xFF, 0xD7, 0), 2, cv2.FONT_HERSHEY_SIMPLEX)
                else:
                    cv2.putText(im0, f'(out): {counter[2]} | (in): {counter[3]}', (
                        10, 50), font, 0.5, (0xFF, 0xD7, 0), 2, cv2.FONT_HERSHEY_SIMPLEX)

            #FPS
            #fps = 1./(time.time()-t1)
            #cv2.putText(im0, "FPS: {:.1f}".format(fps), (0,30), 0, 1, (0,0,255), 2)

            # Showcase results
            if view_frame:
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
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    execution_time = time.time() - t0
    inflow = counter[2]
    outflow = counter[3]

    point_dict = {'ID': '',
                  'Line': '',
                  'Station': '',
                  'Windpass': '',
                  'Direction': '',
                  'Is Access': ''}

    flow_dict = {'Execution Time [s]': execution_time,
                   'Passenger Inflow': inflow,
                   'Passenger Outflow': outflow}

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


if __name__ == '__main__':
    import argparse

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

    frame_failure = 0
    with torch.no_grad():
        try:
            detect(source=opt.source, model=opt.weights, deepsort=opt.deepsort,
                output=opt.output, img_size=opt.img_size, threshold=opt.detection_thr, margin=opt.margin,
                iou_threshold=opt.iou_thr, flip_roi_axis=opt.flip_axis,
                show_axis=opt.show_axis, device=opt.device, trajectory=opt.trajectory,
                fourcc=opt.fourcc, view_frame=opt.view_frame, save_source=opt.save_source,
                agnostic_nms=opt.agnostic_nms, augment=opt.augment)
        except Exception as exp:
            frame_failure += 1
            pass

