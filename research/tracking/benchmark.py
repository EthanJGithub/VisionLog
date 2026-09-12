from pathlib import Path
import cv2,json,time,yaml,ultralytics,torch
from ultralytics import YOLO
import argparse
parser=argparse.ArgumentParser(description='Offline tracker candidate comparison; never publishes portfolio media.')
parser.add_argument('--source',required=True)
parser.add_argument('--start-frame',type=int,default=404)
parser.add_argument('--frames',type=int,default=150)
parser.add_argument('--model-dir',type=Path,default=Path('.'))
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
out=args.output;out.mkdir(parents=True,exist_ok=True)
config=Path(ultralytics.__file__).parent/'cfg/trackers'
for name,model,size,reid in [('baseline','yolo26m.pt',640,False),('tracktrack','yolo26x.pt',960,True),('botsort','yolo26x.pt',960,True)]:
 cfg=yaml.safe_load((config/('bytetrack.yaml' if name=='baseline' else name+'.yaml')).read_text(encoding='utf-8'))
 if reid:cfg.update(with_reid=True,model='auto',track_high_thresh=.4,track_low_thresh=.1,new_track_thresh=.5)
 target=out/(name+'.yaml');target.write_text(yaml.safe_dump(cfg))
 net=YOLO(str(args.model_dir/model));v=cv2.VideoCapture(args.source);v.set(cv2.CAP_PROP_POS_FRAMES,args.start_frame);rows=[];t=time.perf_counter()
 for i in range(args.frames):
  ok,frame=v.read()
  if not ok:break
  result=net.track(frame,persist=True,tracker=str(target),imgsz=size,conf=.1,classes=[0],device=0,verbose=False)[0]
  boxes=result.boxes;d=[]
  if boxes.id is not None:
   for box,confidence,tid in zip(boxes.xyxy.cpu().tolist(),boxes.conf.cpu().tolist(),boxes.id.cpu().tolist()):d.append({'id':int(tid),'xyxy':box,'confidence':confidence})
  rows.append({'frame':i,'detections':d})
  if i%10==0:cv2.imwrite(str(out/f'{name}-{i:03}.jpg'),result.plot());print(name,i,len(d),flush=True)
 (out/(name+'.json')).write_text(json.dumps({'ultralytics':ultralytics.__version__,'model':model,'imgsz':size,'tracker':cfg,'source':args.source,'source_start_frame':args.start_frame,'fps':v.get(cv2.CAP_PROP_FPS),'seconds':len(rows)/v.get(cv2.CAP_PROP_FPS),'elapsed':time.perf_counter()-t,'frames':rows},indent=2))
 print('DONE',name,'unique IDs',len({d['id'] for f in rows for d in f['detections']}),flush=True)
