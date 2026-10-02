"""Read-only B5 search; all outputs go to results/<exp-id>, never retarget/.

MUJOCO_GL=egl uv run python scripts/b5_placement.py --workers 4
Use --phase grid/full/report to resume; grid/full cache exact code + input hashes.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
import json
import os
from pathlib import Path
import time

os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

from wuji_cap.common import frames
from wuji_cap.common.derived_io import read_npz
from wuji_cap.common.joint_order import SIDES
from wuji_cap.common.provenance import REPO_ROOT, provenance
from wuji_cap.sim.g1_model import G1Wuji
from wuji_cap.sim.placement import (Clearance, clearance_arrays, evaluate, leg_pose,
                                    objective, reference_sole_height, summarize_session)
from wuji_cap.sim.render import g1_behind_camera, put
from wuji_cap.sim.scene import add_spheres

SESSIONS = ('session_016','session_017','session_021','session_032')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sessions():
    return {s: read_npz(REPO_ROOT / f'data/derived/retarget/{s}.npz')[0] for s in SESSIONS}


def tag(offset):
    return 'x{:+.4f}_z{:+.4f}'.format(offset[0],offset[2])


def worker(offset, stride, root, fingerprint, baseline=False):
    folder=Path(root) / ('baseline' if baseline else ('grid' if stride else 'full')) / tag(offset)
    folder.mkdir(parents=True, exist_ok=True)
    path=folder/'summary.json'
    if path.exists():
        existing=json.loads(path.read_text())
        if existing.get('fingerprint') == fingerprint:
            return existing
    start=time.monotonic()
    result, outputs=evaluate(offset,load_sessions(),stride,baseline)
    result.update(fingerprint=fingerprint, elapsed_s=time.monotonic()-start,
                  artifact_directory=str(folder.relative_to(REPO_ROOT)))
    for s,arrays in outputs.items():
        np.savez_compressed(folder/(s+'.npz'),**arrays)
    write_json(path,result)
    return result


def pool_evaluate(offsets, stride, root, fingerprint, workers):
    results=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(worker, tuple(o), stride, str(root),fingerprint):o for o in offsets}
        for future in as_completed(pending):
            result=future.result();results.append(result)
            print(json.dumps({'offset':result['offset_m'],'stride':stride,
                              'failures':result['reach_failure_hand_frames'],
                              'collisions':result['body_collision_hand_frames'],
                              'elapsed_s':round(result['elapsed_s'],2)}),flush=True)
    return sorted(results,key=lambda r:tuple(r['offset_m']))


def heatmap(grid, path, chosen=None, full=False):
    xs=sorted(set(r['offset_m'][0] for r in grid));zs=sorted(set(r['offset_m'][2] for r in grid),reverse=True)
    image=np.full((155+90*len(zs),210+155*len(xs),3),250,np.uint8)
    put(image,'B5 global search: reach failures / body collisions (hand-frames)',(15,28),.65)
    subtitle='Every usable frame; x forward, z up. Waist fixed.' if full else 'Subsample includes EVERY existing failure; x forward, z up. Waist fixed.'
    put(image,subtitle,(15,54),.48)
    maxfail=max(1,max(r['reach_failure_hand_frames'] for r in grid))
    lookup={(r['offset_m'][0],r['offset_m'][2]):r for r in grid}
    for ix,x in enumerate(xs):
        put(image,f'dx={x:+.4f} m',(190+155*ix,92),.5)
    for iz,z in enumerate(zs):
        put(image,f'dz={z:+.3f} m',(10,140+90*iz),.5)
        for ix,x in enumerate(xs):
            r=lookup.get((x,z));x0,y0=175+155*ix,110+90*iz
            if r is None:continue
            f,c=r['reach_failure_hand_frames'],r['body_collision_hand_frames']
            color=(int(220-100*f/maxfail),int(245-180*f/maxfail),250)
            cv2.rectangle(image,(x0,y0),(x0+148,y0+80),color,-1)
            put(image,f'{f} / {c}',(x0+10,y0+34),.65)
            put(image,f'p95 {r["p95_position_error_m"]*1000:.2f} mm',(x0+8,y0+61),.4)
            if chosen is not None and np.array_equal(r['offset_m'],chosen):
                cv2.rectangle(image,(x0,y0),(x0+148,y0+80),(220,30,20),3)
    imageio.imwrite(path,image)


def render_stills(root, baseline, chosen, sessions, leg_q=None):
    """All session_032 left failure frames 65..111 and one failure per other session."""
    g=G1Wuji(480,400,pelvis_offset=chosen['offset_m']);base=G1Wuji(480,400)
    T_ref=base.T_world_from_torso() @ frames.T_torso_from_cam()
    with mujoco.Renderer(g.model,400,480) as renderer, mujoco.Renderer(base.model,400,480) as rb:
        panels=[]
        for session,rt in sessions.items():
            failures=np.flatnonzero(rt['left_reach_failure'] | rt['right_reach_failure'])
            if not len(failures):continue
            rows=[int(failures[len(failures)//2])] if session!='session_032' else list(range(65,112))
            before=dict(np.load(REPO_ROOT/baseline['artifact_directory']/(session+'.npz')))
            after=dict(np.load(REPO_ROOT/chosen['artifact_directory']/(session+'.npz')))
            for row in rows:
                views=[]
                for robot,ren,arrays,label in ((base,rb,before,'Before'),(g,renderer,after,'After')):
                    robot.data.qpos[:]=robot.q_home if robot is base or leg_q is None else leg_q
                    t=int(np.flatnonzero(arrays['rows']==row)[0])
                    for s in SIDES:
                        robot.data.qpos[robot.arm_qpos_adr[s]]=arrays[s+'_q_arm_hold'][t]
                        robot.hands[s].set_q_sdk(robot.data,rt[s+'_q_wuji_cmd'][row] if not rt[s+'_valid'][row] else rt[s+'_q_wuji'][row])
                    mujoco.mj_forward(robot.model,robot.data)
                    camera=g1_behind_camera(robot)
                    camera.distance=1.5
                    camera.lookat[2]+=0.12
                    ren.update_scene(robot.data,camera)
                    for s in SIDES:
                        if rt[s+'_valid'][row]:
                            target=(T_ref @ rt[s+'_T_cam_from_palm'][row])[:3,3]
                            add_spheres(ren.scene,target[None],(.1,.85,.2,1),.006)
                            add_spheres(ren.scene,robot.data.xpos[robot.palm_body_ids[s]][None],(.9,.15,.15,1),.004)
                    img=ren.render().copy()
                    cv2.rectangle(img,(0,0),(480,62),(25,25,25),-1)
                    cv2.rectangle(img,(0,338),(480,400),(25,25,25),-1)
                    put(img,f'{label}: {session} frame {rt["frame_index"][row]}',(10,22),.55,(250,250,250),2)
                    put(img,'L',(10,55),.7,(250,250,250),2);put(img,'R',(450,55),.7,(250,250,250),2)
                    for k,s in enumerate(SIDES):
                        if rt[s+'_valid'][row]:
                            put(img,f'{s}: err {arrays[s+"_pos_err_m"][t]*1000:.1f} mm, clear {arrays[s+"_clearance_m"][t]*1000:.1f} mm',
                                (10,355+23*k),.42,(250,250,250),1)
                    views.append(img)
                pair=np.concatenate(views,axis=1)
                imageio.imwrite(root/'stills'/f'{session}_{row:04d}.png',pair)
                if session!='session_032' or row in (65,88,111):panels.append(pair)
        imageio.imwrite(root/'g1_stills.png',np.concatenate(panels,axis=0))
    # Full-body view explicitly displays the leg pose and floor.
    if leg_q is not None:
        g.data.qpos[:]=leg_q;mujoco.mj_forward(g.model,g.data)
        cam=mujoco.MjvCamera();cam.lookat[:]=[chosen['offset_m'][0],0,.65];cam.distance=2.3;cam.azimuth=130;cam.elevation=-15
        with mujoco.Renderer(g.model,400,480) as ren:
            ren.update_scene(g.data,cam);img=ren.render().copy()
            cv2.rectangle(img,(0,0),(480,35),(25,25,25),-1)
            put(img,'Leg feasibility pose (kinematic only)',(10,23),.5,(255,255,255),1)
            imageio.imwrite(root/'leg_pose.png',img)


def report(root, grid, full, baseline, meta):
    sessions=load_sessions();chosen=min(full,key=objective)
    feasible=chosen['reach_failure_hand_frames']==0 and chosen['body_collision_hand_frames']==0
    per_session={}
    def session_p95(candidate, session):
        with np.load(REPO_ROOT/candidate['artifact_directory']/(session+'.npz')) as arrays:
            errors=np.concatenate([arrays[side+'_pos_err_m'] for side in SIDES])
        return float(np.percentile(errors[np.isfinite(errors)],95))
    if not feasible:
        for s in SESSIONS:
            def score(r):
                vals=list(r['sessions'][s].values())
                return (sum(v['reach_failures'] for v in vals),sum(v['body_collision_frames'] for v in vals),
                        session_p95(r, s),tuple(r['offset_m']))
            best=min(full,key=score)
            robot=G1Wuji(pelvis_offset=best['offset_m'])
            q_session, legs_session=leg_pose(robot,float(robot.model.geom('floor').pos[2]))
            validated=None
            if legs_session['feasible']:
                robot.q_home[:]=q_session
                with np.load(REPO_ROOT/best['artifact_directory']/(s+'.npz')) as saved:
                    arrays={k:saved[k] for k in saved.files}
                clearance_arrays(robot,sessions[s],arrays['rows'],arrays)
                validated=summarize_session(sessions[s],arrays['rows'],arrays)
                np.savez_compressed(root/'leg_validation'/(s+'_per_session.npz'),**arrays)
            per_session[s]={'offset_m':best['offset_m'],'metrics':best['sessions'][s],
                            'p95_position_error_m':session_p95(best,s),
                            'head_camera_displacement_m':best['head_camera_displacement_m'],
                            'leg_pose_at_xml_floor':legs_session,
                            'clearance_with_grounded_legs':validated,
                            'artifact_directory':best['artifact_directory']}
    g=G1Wuji(pelvis_offset=chosen['offset_m']);base=G1Wuji()
    datum=reference_sole_height(base)
    q_reference,reference_legs=leg_pose(g,datum)
    q_floor,floor_legs=leg_pose(g,float(g.model.geom('floor').pos[2]))
    lower=chosen['offset_m'][2]<0
    leg_q=q_floor if lower and floor_legs['feasible'] else (q_reference if lower and reference_legs['feasible'] else None)
    # Recheck body clearance using the feasible legs as they change the knee geometry.
    leg_clearance={}
    if leg_q is not None:
        g.q_home[:]=leg_q
        for session,rt in sessions.items():
            arrays=dict(np.load(REPO_ROOT/chosen['artifact_directory']/(session+'.npz')))
            clearance_arrays(g,rt,arrays['rows'],arrays)
            leg_clearance[session]=summarize_session(rt,arrays['rows'],arrays)
            np.savez_compressed(root/'leg_validation'/(session+'.npz'),**arrays)
    for p in (root/'stills',):p.mkdir(exist_ok=True)
    render_stills(root,baseline,chosen,sessions,leg_q)
    heatmap(grid,root/'placement_grid.png',chosen['offset_m'])
    heatmap(full,root/'placement_full.png',chosen['offset_m'],full=True)
    inputs_after={s:sha(REPO_ROOT/f'data/derived/retarget/{s}.npz') for s in SESSIONS}
    if inputs_after!=meta['input_sha256']:raise RuntimeError('shared retarget inputs changed during B5')
    metrics={**meta,'assumptions':['AS-01','AS-02','AS-09','AS-12','AS-B08'],
             'kinematic_only':True,'target_frame':'fixed original home-pose camera',
             'objective_order':['reach_failure_hand_frames','body_collision_hand_frames','p95_position_error_m'],
             'collision_scope':'existing convex collision geoms: Wuji hands and G1 elbows/wrists vs trunk/head/pelvis/legs; valid contract frames only',
             'collision_geometry':Clearance(base).inventory(),
             'thresholds':{'position_error_m':.01,'rotation_error_deg':5,'body_collision_distance_m':0},
             'baseline':baseline,'grid':grid,'full_candidates':full,'chosen_global':chosen,
             'global_zero_failures_zero_collisions':feasible,'best_per_session':per_session,
             'grounded_body_collision_hand_frames':sum(v['body_collision_frames'] for ss in leg_clearance.values() for v in ss.values()) if floor_legs['feasible'] else None,
             'placement_meets_full_target':bool(feasible and (not lower or floor_legs['feasible']) and all(v['body_collision_frames']==0 for ss in leg_clearance.values() for v in ss.values())),
             'recommendation_status':'PROPOSED; best tested offset, not a zero-failure solution' if not feasible else 'PROPOSED',
             'numerical_run_provenance':json.loads((root/'coarse/run_metadata.json').read_text()) if (root/'coarse/run_metadata.json').exists() else meta,
             'leg_feasibility':{'home_sole_z_m':datum,'xml_floor_z_m':float(base.model.geom('floor').pos[2]),
                                'same_home_sole_plane':reference_legs,'xml_floor':floor_legs,
                                'clearance_with_feasible_legs':leg_clearance,
                                'selected_qpos':leg_q.tolist() if leg_q is not None else None},
             'input_sha256_after':inputs_after,'inputs_unchanged':True,
             'limitations':['Finite grid and existing IK solver: no proof of global geometric infeasibility.',
                            'Collision checks exclude hand-hand, arm-arm, upper-arm/body, objects and interpolated motion.',
                            'Existing mesh collision approximations and camera/mount assumptions apply.',
                            'Home keyframe is elevated above the XML floor. Flat-foot feasibility is reported for both planes.',
                            'No balance, dynamics or hardware claim.']}
    write_json(root/'metrics.json',metrics)
    print(json.dumps({'chosen_global':chosen['offset_m'],'failures':chosen['reach_failure_hand_frames'],
                      'collisions':chosen['body_collision_hand_frames'],'legs_at_xml_floor':floor_legs['feasible']}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exp-id',default='20261001-b5-placement')
    p.add_argument('--phase',choices=('all','grid','full','report'),default='all')
    p.add_argument('--dx',type=float,nargs='+',default=[0,.1,.2,.3])
    p.add_argument('--dz',type=float,nargs='+',default=[0,-.1,-.2,-.3])
    p.add_argument('--stride',type=int,default=12);p.add_argument('--top',type=int,default=3)
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    if Path(args.exp_id).name!=args.exp_id:raise ValueError('exp-id must be a directory name')
    root=REPO_ROOT/'results'/args.exp_id;root.mkdir(parents=True,exist_ok=True)
    meta={**provenance('scripts/b5_placement.py'),
          'input_sha256':{s:sha(REPO_ROOT/f'data/derived/retarget/{s}.npz') for s in SESSIONS},
          'params':vars(args)}
    fingerprint=hashlib.sha256(json.dumps([meta['code_sha256'],meta['input_sha256'],args.stride],sort_keys=True).encode()).hexdigest()
    write_json(root/'run_metadata.json',meta)
    offsets=[(x,0,z) for x,z in itertools.product(args.dx,args.dz)]
    if args.phase in ('all','grid'):
        grid=pool_evaluate(offsets,args.stride,root,fingerprint,args.workers)
        write_json(root/'grid.json',grid);heatmap(grid,root/'placement_grid.png')
    else:grid=json.loads((root/'grid.json').read_text())
    if args.phase=='grid':return
    # Best few globally plus the best per session are fully evaluated, so per-session
    # fallback never relies on subsampled statistics.
    ranked=sorted(grid,key=objective);selected={tuple(r['offset_m']) for r in ranked[:args.top]}
    if not any(r['reach_failure_hand_frames']==r['body_collision_hand_frames']==0 for r in ranked):
        for session in SESSIONS:
            def session_score(r):
                v=list(r['sessions'][session].values())
                return (sum(x['reach_failures'] for x in v),sum(x['body_collision_frames'] for x in v),
                        max(x['p95_position_error_m'] or 0 for x in v),tuple(r['offset_m']))
            selected.add(tuple(min(grid,key=session_score)['offset_m']))
    if args.phase in ('all','full'):
        baseline=worker((0,0,0),None,str(root),fingerprint,True)
        full=pool_evaluate(sorted(selected),None,root,fingerprint,args.workers)
        write_json(root/'full.json',full);write_json(root/'baseline.json',baseline)
    else:
        full=json.loads((root/'full.json').read_text());baseline=json.loads((root/'baseline.json').read_text())
    if args.phase=='full':return
    (root/'leg_validation').mkdir(exist_ok=True)
    report(root,grid,full,baseline,meta)


if __name__=='__main__':main()
