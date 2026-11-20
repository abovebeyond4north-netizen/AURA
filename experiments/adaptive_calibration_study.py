"""Matched public-seed comparison of joint calibration and existing controls."""
import argparse, json
from pathlib import Path
from experiments.persistent_pose_study import trial
from aura.exploration.adaptive_calibration import CalibratedComposedCamera, CalibratedPersistentCamera
from aura.exploration.composed_routes import ComposedRouteCamera
from aura.exploration.persistent_pose import PersistentPoseCamera

def audited_trial(seed,size,col,rev,noise,cls):
    completed=[]
    class AuditedCalibration(cls):
        def after_key_pickup(self):
            super().after_key_pickup()
            completed.append(bool(self.reversed))
    AuditedCalibration.__name__=cls.__name__
    row=trial(seed,size,col,rev,noise,AuditedCalibration)
    row['completed_calibrations']=len(completed)
    row['mapping_errors']=sum(x!=rev for x in completed)
    return row

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--layouts',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--noise',default='0,.03,.08');a=p.parse_args()
    rows=[audited_trial(a.seed,size,col,rev,noise,cls) for size,col in [tuple(map(int,x.split(':'))) for x in a.layouts.split(',')] for rev in (False,True) for noise in map(float,a.noise.split(',')) for cls in (PersistentPoseCamera,ComposedRouteCamera,CalibratedPersistentCamera,CalibratedComposedCamera)]
    with a.output.open('x') as f:json.dump(rows,f,indent=2)
if __name__=='__main__':main()
