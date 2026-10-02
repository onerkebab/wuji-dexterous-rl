import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from wuji_cap.common import frames
from wuji_cap.common.joint_order import SIDES
from wuji_cap.sim.arm_ik import ArmIK
from wuji_cap.sim.g1_model import G1Wuji, g1_xml_path
from wuji_cap.sim.placement import Clearance, evaluate, leg_pose, objective, reference_sole_height, sample_rows

pytestmark=pytest.mark.skipif(not g1_xml_path().is_file(),reason='TWIST2 G1 model not found')
ROOT=Path(__file__).resolve().parents[2]


def test_default_bitwise_prechange_session032_slice():
    raw=json.loads((ROOT/'results/20261001-b5-placement/default_regression_fixture.json').read_text())
    a={k:np.asarray(v) for k,v in raw['arrays'].items()}
    result=ArmIK().run({s:a[s+'_T_cam_from_palm'] for s in SIDES},{s:a[s+'_valid'] for s in SIDES})
    for s in SIDES:
        for k in ('q_arm','pos_err_m','rot_err_rad','n_iters','start_used','q_arm_hold'):
            np.testing.assert_array_equal(getattr(result,k)[s],a[s+'_'+k])


def test_offset_and_reference_camera_keep_world_targets_fixed():
    a=G1Wuji();b=G1Wuji(pelvis_offset=(.12,0,-.08))
    np.testing.assert_array_equal(b.q_home[3:],a.q_home[3:])
    np.testing.assert_allclose(b.q_home[:3]-a.q_home[:3],[.12,0,-.08],atol=1e-15)
    np.testing.assert_array_equal(b.home_qpos(),b.q_home)
    T_ref=ArmIK(a).T_world_from_cam
    ik=ArmIK(b,T_world_from_cam=T_ref)
    T_ref[0,3]+=1  # input copied, not aliased
    np.testing.assert_array_equal(ik.T_world_from_cam,ArmIK(a).T_world_from_cam)
    moved=ArmIK(b).T_world_from_cam
    np.testing.assert_allclose(moved[:3,3]-ik.T_world_from_cam[:3,3],[.12,0,-.08])
    ik.config.update(b.q_home)
    target=frames.inv_T(ik.T_world_from_cam) @ b.body_T(b.palm_body_ids['left'],ik.config.data)
    result=ik.run({s:target[None] for s in SIDES},{'left':np.array([True]),'right':np.array([False])})
    assert result.pos_err_m['left'][0]<1e-10
    np.testing.assert_allclose(result.T_world_from_palm_target['left'][0],ik.T_world_from_cam @ target)


@pytest.mark.parametrize('offset',[(0,0),[0,np.nan,0],[0,0,0,0]])
def test_invalid_offset(offset):
    with pytest.raises(ValueError):G1Wuji(pelvis_offset=offset)


def test_invalid_camera_transform():
    with pytest.raises(ValueError):ArmIK(T_world_from_cam=np.zeros((4,4)))


def test_clearance_detects_forced_hand_in_torso_and_matches_brute_force():
    g=G1Wuji();check=Clearance(g);m,d=g.model,g.data
    # Place the Wuji wrist at the torso collision mesh centre by changing its mount.
    palm=g.palm_body_ids['left'];parent=m.body_parentid[palm]
    body_geom=next(i for i in check.body if m.geom_dataid[i]>=0 and m.mesh(m.geom_dataid[i]).name=='torso_link')
    target=d.geom_xpos[body_geom].copy()
    R=d.xmat[parent].reshape(3,3)
    m.body_pos[palm]=R.T @ (target-d.xpos[parent])
    mujoco.mj_kinematics(m,d)
    minimum,pair=check.minimum(d,'left')
    assert minimum<0, (minimum,pair)
    brute=min(mujoco.mj_geomDistance(m,d,int(a),int(b),10.,None) for a,b in check.pairs['left'])
    assert minimum==brute
    # Positive clearance uses the same exact pair-distance minimum.
    g2=G1Wuji();check2=Clearance(g2)
    for s in SIDES:
        brute=min(mujoco.mj_geomDistance(g2.model,g2.data,int(a),int(b),10.,None) for a,b in check2.pairs[s])
        assert check2.minimum(g2.data,s)[0]==brute


def test_sample_keeps_all_failures_and_excludes_invalid():
    rt={s+'_valid':np.ones(13,bool) for s in SIDES}
    rt.update({s+'_reach_failure':np.zeros(13,bool) for s in SIDES})
    rt['left_reach_failure'][[2,3,9]]=True
    rt['right_reach_failure'][7]=True
    rt['left_valid'][11]=rt['right_valid'][11]=False
    rows=sample_rows(rt,5)
    assert {2,3,7,9}<=set(rows) and 11 not in rows
    np.testing.assert_array_equal(rows,np.unique(rows))
    with pytest.raises(ValueError):sample_rows(rt,0)


def test_search_deterministic_and_masks_gaps():
    ik=ArmIK();g=ik.g;rt={'frame_index':np.arange(3),'timestamps_s':np.arange(3)/30}
    for s in SIDES:
        target=frames.inv_T(ik.T_world_from_cam) @ g.body_T(g.palm_body_ids[s])
        rt[s+'_T_cam_from_palm']=np.repeat(target[None],3,axis=0)
        rt[s+'_valid']=np.array([True,False,True]);rt[s+'_reach_failure']=np.zeros(3,bool)
        rt[s+'_q_wuji']=np.zeros((3,20));rt[s+'_q_wuji'][1]=np.nan
        rt[s+'_q_wuji_cmd']=np.zeros((3,20))
    a,aa=evaluate((0,0,0),{'synthetic':rt});b,bb=evaluate((0,0,0),{'synthetic':rt})
    assert a==b
    for k in aa['synthetic']:np.testing.assert_array_equal(aa['synthetic'][k],bb['synthetic'][k])
    assert a['usable_hand_frames']==4 and a['reach_failure_hand_frames']==0
    assert np.isnan(aa['synthetic']['left_clearance_m'][1])
    assert not aa['synthetic']['left_body_collision'][1]
    colliding={**a,'body_collision_hand_frames':1}
    failing={**a,'reach_failure_hand_frames':1}
    assert objective(a)<objective(colliding)<objective(failing)


def test_lowered_legs_flat_and_within_limits_at_reference_sole_plane():
    g=G1Wuji(pelvis_offset=(.1,0,-.1));q,info=leg_pose(g,reference_sole_height(G1Wuji()))
    assert info['feasible']
    d=mujoco.MjData(g.model);d.qpos[:]=q;mujoco.mj_kinematics(g.model,d)
    for s in SIDES:
        for name,value in info['sides'][s]['joint_rad'].items():
            lo,hi=g.model.joint(name).range
            assert lo<=value<=hi
        R=d.xmat[g.model.body(s+'_ankle_roll_link').id].reshape(3,3)
        np.testing.assert_allclose(R,np.eye(3),atol=1e-6)
    np.testing.assert_array_equal(q[g.waist_qpos_adr],g.q_home[g.waist_qpos_adr])
    np.testing.assert_array_equal(q[:7],g.q_home[:7])
