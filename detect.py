import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

#Import necessary libraries
from flask import Flask, render_template, stream_with_context, request, Response
import time
import csv
import pandas as pd
import numpy as np
import plotly.express as px
import plotly
import json
import os


def aconcr(data):
  fig = px.line(data, x = 'sec', y = 'aconcr', title= "누적 집중도")
  graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
  return graphJSON

def concrh(data):
  fig =  px.line(data, x = 'sec', y = 'concrh', title= "시간별 집중도")
  graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
  return graphJSON

def pie(data):
  acc_df = pd.DataFrame(columns=['핸드폰', '졸음', '자리 비움', '고개 돌림'])
  acc_df.loc['hi'] = [data['phone'].sum(), (data['eye_closed'].sum()/2), np.isnan(data['face']).sum(), (data['side'].sum())/2]
  fig = px.pie(acc_df.T, values = 'hi', names = ['핸드폰', '졸음', '자리 비움', '고개 돌림'], title= "집중 방해 요인")
  graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
  return graphJSON

def bar(data):
  acc_df = pd.DataFrame(columns=['핸드폰', '졸음', '자리 비움', '고개 돌림', '집중 시간'])
  acc_df.loc['hi'] = [data['phone'].sum(), (data['eye_closed'].sum() / 2), np.isnan(data['face']).sum(), (data['side'].sum()) / 2, data['conc'].sum()]
  fig = px.bar(acc_df, x = ['핸드폰', '졸음', '자리 비움', '고개 돌림', '집중 시간'] , y = [acc_df['핸드폰']['hi'], acc_df['졸음']['hi'], acc_df['자리 비움']['hi'],acc_df['고개 돌림']['hi'],  acc_df['집중 시간']['hi']], color=acc_df.columns, title = "시간 분석")
  graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
  return graphJSON

def mean_aconcr(csv_list, csv_dic):
  sum1 = 0

  for cs in csv_list:
    data = csv_dic[cs]
    sum1 += (data['aconcr'][data.shape[0]-1])

  return sum1 / len(csv_list)


#Initialize the Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")
rows2=[]
s, s2 = '', ''



def detect(save_img=False):
    global rows2
    global s
    global s2
    global state
    print("in detect...................................")
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    # save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    
    f = open("csv/3.csv", "w")
    fieldnames  = ['sec', 'eye_opened','eye_opened_bbox','eye_closed','eye_closed_bbox', 'face','face_bbox','phone', 'phone_bbox', 'side']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    start = time.time()

    eye_queue = list([0 for _ in range(0, 100)])
    sleep_mode = False

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                # p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                p, s, im0, frame = path[i], '', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # add
                allow_class= ['eye_opened', 'eye_closed', 'face', 'phone']
                eye_closed_num = 0
                eye_opened_num = 0
                phone_num = 0
                left_flag = True
                face = []
                left_eye = []
                right_eye = []

                # Print results
                for c in det[:, -1]:  # detections per class
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    
                    if 'eye_opened' in names[int(c)]:
                      eye_opened_num += 1
                    if 'eye_closed' in names[int(c)]:
                      eye_closed_num += 1
                    if 'phone' in names[int(c)]:
                      phone_num += 1

                if phone_num >= 1:
                  cv2.putText(im0, "phone", (5, 80), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))

                elif eye_closed_num == 2:
                  eye_queue.pop()
                  eye_queue.insert(0, 1)
                
                else:
                  eye_queue.pop()
                  eye_queue.insert(0, 0)

                sum_queue= sum(eye_queue)
                
                if sum_queue >= 90 and sleep_mode==False:
                    # csv 기록하기
                    cv2.putText(im0, "eye_closed", (5, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))
                    sleep_mode = True
                elif sum_queue >= 90 and sleep_mode==True:
                    cv2.putText(im0, "eye_closed", (5, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))
                    sleep_mode = True
                elif sum_queue < 90: 
                    cv2.putText(im0, "focus", (5, 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))
                    sleep_mode = False

                # Write results
                csv_dict = {}
                for *xyxy, conf, cls in reversed(det):
                  if names[int(cls)] in allow_class:
                    if names[int(cls)] in csv_dict:
                      csv_dict[f'{names[int(cls)]}'] += 1
                      csv_dict[f'{names[int(cls)]}_bbox'].append(list(map(int, xyxy)))
                    else:
                      csv_dict[f'{names[int(cls)]}'] = 1
                      csv_dict[f'{names[int(cls)]}_bbox'] = [list(map(int, xyxy))]

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        c = int(cls)
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        

                        if eye_opened_num == 1 and eye_closed_num == 0:
                          csv_dict['side'] = 1
                          cv2.putText(im0, "side", (5, 110), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))

                        elif eye_opened_num == 2:
                          if c == 4:
                            face = [xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()]
                          if c == 0 and left_flag == True: # left_eye
                            left_eye = [xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()]
                            left_flag = False
                          if c == 0 and left_flag == False: # right_eye
                            right_eye = [xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()]
                            left_flag = True
        
                          lr_threshold = 13.0 # optimal = 13.0

                          if face and left_eye and right_eye:
                            if left_eye[0] - face[0] < lr_threshold:
                              csv_dict['side'] = 1
                              cv2.putText(im0, "side", (5, 110), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))

                            if face[2] - right_eye[2] < lr_threshold:
                              csv_dict['side'] = 1
                              cv2.putText(im0, "side", (5, 110), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))

            csv_dict['sec'] = round(time.time() - start, 2)
            try:
              writer.writerow(csv_dict)
            except Exception as exc:
              exc.args += (csv_dict,)
              raise            
            # Print time (inference + NMS)
            if s == '':
                s = 'Empty.'
            else:
                s = s[:-2] + '.'
            #print(f'{s}Done. ({t2 - t1:.3f}s)')
            ret, im0 = cv2.imencode('.jpg', im0)
            frame = im0.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='model2.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()

def out():
    global s2
    global s
    while(True):
        if s2 != s and s.endswith("."):
            s2 = s
            yield s2+"<br>"
        else:
            yield ''


def stream_template(template_name, **context):
    app.update_template_context(context)
    t = app.jinja_env.get_template(template_name)
    rv = t.stream(context)
    rv.disable_buffering()
    return rv


# flask app code

@app.route('/', methods = ['POST', 'GET'])
def stream_view():
    global rows2
    rows = []
    for i in range(10):
      rows.append(str(i))
    # rows = rows2
    return Response(stream_with_context(stream_template('index.html', rows=rows2)))

@app.route('/video_feed')
def video_feed():
    return Response(detect(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/output')
def output():
    return Response(out(), mimetype='text/html')

@app.route('/1')
def first():
    return render_template('graph.html', graphJSON = {'concrh': concrh(csv_dic['1.csv']), 'aconcr': aconcr(csv_dic['1.csv']), 'pie': pie(csv_dic['1.csv']), 'bar': bar(csv_dic['1.csv'])})
@app.route('/2')
def second():
    return render_template('graph.html', graphJSON = {'concrh': concrh(csv_dic['2.csv']), 'aconcr': aconcr(csv_dic['2.csv']), 'pie': pie(csv_dic['2.csv']), 'bar': bar(csv_dic['2.csv'])})
@app.route('/3')
def third():
  
    data = pd.read_csv('csv/3.csv')

    data['conc'] = 1.0

    for i in range(data.shape[0]):
        if np.isnan(data['eye_opened'][i]):
            data['conc'][i] = 0.0
        elif np.isnan((data['face'][i])):
            data['conc'][i] = 0.0
        elif not (np.isnan(data['phone'][i])):
            data['conc'][i] = 0.0
        elif not (np.isnan(data['side'][i])):
            data['conc'][i] = 0.5

    data['aconc'] = data['conc']
    data['aconcr'] = data['aconc']
    data['concrh'] = data['conc']
    
    k = 0.6

    for i in range(1, data.shape[0]):
        data['aconc'][i] = data['aconc'][i - 1] + data['conc'][i]
        data['aconcr'][i] = (data['aconc'][i]) / (i + 1)
      
        if i > 20:
            sum1 = 0
            for j in range(20):
                sum1 += (20 - j) * data['conc'][i - j]

            data['concrh'][i] = sum1 / 210

    csv_dic[cs] = data

    return render_template('graph.html', graphJSON = {'concrh': concrh(csv_dic['3.csv']), 'aconcr': aconcr(csv_dic['3.csv']), 'pie': pie(csv_dic['3.csv']), 'bar': bar(csv_dic['3.csv'])})

if __name__ == "__main__":
    path = 'csv/'

    csv_list = os.listdir(path)
    for cs in csv_list:
        if cs[-3:] != 'csv':
            csv_list.remove(cs)

    csv_dic = {}

    for cs in csv_list:

        data = pd.read_csv(path + cs)

        data['conc'] = 1.0

        for i in range(data.shape[0]):
            if np.isnan(data['eye_opened'][i]):
                data['conc'][i] = 0.0
            elif np.isnan((data['face'][i])):
                data['conc'][i] = 0.0
            elif not (np.isnan(data['phone'][i])):
                data['conc'][i] = 0.0
            elif not (np.isnan(data['side'][i])):
                data['conc'][i] = 0.5

        data['aconc'] = data['conc']
        data['aconcr'] = data['aconc']
        data['concrh'] = data['conc']
        k = 0.6

        for i in range(1, data.shape[0]):
            data['aconc'][i] = data['aconc'][i - 1] + data['conc'][i]
            data['aconcr'][i] = (data['aconc'][i]) / (i + 1)
            if i > 20:
                sum1 = 0
                for j in range(20):
                    sum1 += (20 - j) * data['conc'][i - j]

                data['concrh'][i] = sum1 / 210

        csv_dic[cs] = data

    app.run(host='localhost', debug=True)            

