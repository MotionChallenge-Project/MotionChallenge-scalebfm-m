"""Independent author ScaleBridge history and feature comparison for raw M weights."""
import ast, json, sys, hashlib, argparse
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
A=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
for name in ['source','weights','metadata','mode-table','output']:parser.add_argument('--'+name,type=Path,required=True)
args=parser.parse_args();S=args.source.resolve()
pins=json.loads((A/'native-sources.json').read_text())
for name,digest in pins['files'].items():assert hashlib.sha256((S/name).read_bytes()).hexdigest()==digest,name
sys.path.insert(0,str(A))
from adapter import ScaleBFM
from native_runtime import load_policy
from motion_challenge.sdk import StatePacket, ReferenceFrame

def definitions(path,names,scope,methods=None):
 nodes=[x for x in ast.parse(path.read_text()).body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name in names]
 assert {x.name for x in nodes}==set(names)
 for n in nodes:
  n.decorator_list=[]
  if isinstance(n,ast.ClassDef):
   n.bases=[]
   if methods:n.body=[x for x in n.body if isinstance(x,ast.FunctionDef) and x.name in methods]
   for x in n.body:
    if isinstance(x,ast.FunctionDef):x.decorator_list=[]
 exec(compile(ast.Module(nodes,[]),str(path),'exec'),scope)

runtime=json.loads((A/'runtime.json').read_text());runtime['weights']=str(args.weights.resolve())
paths={'network':S/'ScaleTrack/source/my_rsl_rl/my_rsl_rl/networks/humanoid_transformer.py','export':S/'ScaleTrack/scripts/pretrain/rsl_rl/play_export_check_humanoid_transformer.py','robot':S/'ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof/g1_29dof.xml','metadata':args.metadata.resolve(),'mode_table':args.mode_table.resolve()}
for k,p in paths.items():runtime['artifacts'][k]['path']=str(p)
policy=ScaleBFM(A,runtime);native_policy,meta,native=load_policy(paths,runtime['weights'])
source=S/'ScaleBridge/scalebridge';scope={'torch':torch,'Tensor':torch.Tensor,'Dict':dict,'logger':NS(info=lambda *_:None)}
definitions(source/'utils/torch_utils.py',['quat_apply_inverse','quat_mul_inverse_left'],scope)
funcs=['root_quat_buffer','base_ang_vel_buffer','dof_pos_buffer','dof_vel_buffer','actions_buffer','target_body_pos_future_to_robot_base','target_body_rot_future_to_robot_base','future_time_offsets']
definitions(source/'observation/components.py',funcs,scope)
definitions(source/'env/base_env.py',['BaseEnv'],scope,{'_setup_state_buffer','_setup_state_manager','_update_state_manager','reset'})
definitions(source/'env/motion_tracking.py',['MotionTrackingEnv'],scope,{'_global_time_indices','_gather_reference_state'})
base=scope['BaseEnv'];motion=scope['MotionTrackingEnv'];rng=np.random.default_rng(241024)
errors={k:0. for k in ['reference_fk_position','reference_fk_rotation','history','observation','action','motor_target']}
def compare(k,a,b):
 if isinstance(a,torch.Tensor):a=a.detach().numpy()
 if isinstance(b,torch.Tensor):b=b.detach().numpy()
 errors[k]=max(errors[k],float(np.max(np.abs(np.asarray(a)-np.asarray(b)))))
bodies,joints,parents,axes,translation,rotation=native.parse_xml(str(paths['robot']));rotation/=rotation.norm(dim=-1,keepdim=True)
xml_order=[meta['joint_names'].index(n) for n in joints];ids=[bodies.index(n) for n in meta['selected_body_names']]
def author_fk(frame,order):
 q=torch.tensor(np.asarray(frame.joint_position_rad)[order],dtype=torch.float32)
 half=q[xml_order,None]/2;jrot=torch.cat([torch.cos(half),axes*torch.sin(half)],-1)
 pos=torch.zeros(30,3);quat=torch.zeros(30,4);quat[:,0]=1
 pos[0]=torch.tensor(frame.root_position_world_m);quat[0]=torch.tensor(np.asarray(frame.root_orientation_xyzw)[[3,0,1,2]])
 for j in range(1,30):
  parent=parents[j];pos[j]=pos[parent]+native.quat_apply(quat[parent:parent+1],translation[j-1:j])[0]
  quat[j]=native.quat_mul(quat[parent:parent+1],native.quat_mul(rotation[j-1:j],jrot[j-1:j]))[0]
 return pos[ids],quat[ids]

with torch.inference_mode():
 for reset in range(4):
  names=tuple(np.asarray(meta['joint_names'])[rng.permutation(29)]);order=[names.index(n) for n in meta['joint_names']]
  q0=np.asarray(meta['default_dof_pos']);inverse=[meta['joint_names'].index(n) for n in names]
  frames=[]
  for i in range(38):
   t=i*.02;q=(q0+.1*np.sin(t*3+np.arange(29)*.1))[inverse]
   rot=R.from_euler('xyz',[.08*np.sin(t),.06*np.cos(t),.3*t+reset*.5])
   frames.append(ReferenceFrame(t,t,tuple(q),(t*.2,.05*np.sin(t),.8+.02*np.cos(t)),tuple(rot.as_quat())))
  fk=[author_fk(f,order) for f in frames]
  bodypos=torch.stack([x[0] for x in fk]);bodyquat=torch.stack([x[1] for x in fk])
  env=base();env.metadata_dict=meta;env.device='cpu';env.cfg=NS();env.episode_length_buf=torch.zeros(1,dtype=torch.long)
  current={}
  env.simulator=NS(refresh_sim=lambda:current)
  env._setup_state_manager()
  env._calibrate=lambda:None;env._update_observation_manager=lambda:env.state_buffer
  env.body_pos_w=bodypos;env.body_quat_w=bodyquat;env.motion_len=len(frames);env.future_frame_offset=torch.tensor(meta['future_idx']);env.reference_forcing=True
  env._global_time_indices=motion._global_time_indices.__get__(env)
  for step in range(40):
   idx=min(step,len(frames)-1);fr=frames[idx];actual=R.from_quat(fr.root_orientation_xyzw)*R.from_rotvec(rng.normal(0,.01,3))
   state=StatePacket(step*.02,names,tuple(np.asarray(fr.joint_position_rad)+rng.normal(0,.015,29)),tuple(rng.normal(0,.2,29)),tuple(np.asarray(fr.root_position_world_m)+rng.normal(0,.02,3)),tuple(actual.as_quat()),(0.,)*3,tuple(actual.apply(rng.normal(0,.2,3))))
   refs=tuple(frames[min(step+i,len(frames)-1)] for i in range(6))
   current.clear();current.update(root_pos=torch.tensor(state.root_position_world_m)[None],root_quat_wxyz=torch.tensor(np.asarray(state.root_orientation_xyzw)[[3,0,1,2]],dtype=torch.float32)[None],base_ang_vel=torch.tensor(actual.inv().apply(state.root_angular_velocity_world_rad_s),dtype=torch.float32)[None],dof_pos=torch.tensor(np.asarray(state.joint_position_rad)[order],dtype=torch.float32)[None],dof_vel=torch.tensor(np.asarray(state.joint_velocity_rad_s)[order],dtype=torch.float32)[None])
   if step==0:
    env.reset();motion._gather_reference_state(env)
    # Native BaseEnv.reset fills actual root state after setup gathered reference.
    env.state_buffer['root_pos_buffer'].copy_(current['root_pos'][:,None].expand_as(env.state_buffer['root_pos_buffer']))
    policy.reset(state,refs,reset)
   else:
    env.episode_length_buf[:]=step;env._update_state_manager();motion._gather_reference_state(env)
   for frame in refs:
    p,q=policy.reference_bodies(frame);np_,nq=author_fk(frame,order)
    compare('reference_fk_position',p,np_);compare('reference_fk_rotation',R.from_quat(q[:,[1,2,3,0]]).as_matrix(),R.from_quat(nq.numpy()[:,[1,2,3,0]]).as_matrix())
   native_inputs=[scope[k](env.state_buffer) for k in funcs[:-1]]+[torch.tensor([7]),scope['future_time_offsets'](env.state_buffer)]
   captured=policy.observations(state,refs)
   for a,b in zip(captured,native_inputs):compare('observation',a,b)
   for k in env.state_buffer:
    if k in policy.buffers:compare('history',policy.buffers[k],env.state_buffer[k])
   target,action=native_policy(*native_inputs)
   original=policy.observations;policy.observations=lambda *_:captured
   command=policy.act(state,refs);policy.observations=original
   compare('action',policy.last,action);compare('motor_target',np.asarray(command.joint_position_target_rad),target[0].numpy()[inverse])
   env.action.copy_(action)
report={'status':'passed' if max(errors.values())<2e-5 else 'failed','cases':160,'resets':4,'max_abs_errors':errors,'tolerance':2e-5,'control_mode':7,'reference_forcing':True,'checkpoint_sha256':runtime['weights_sha256'],'source_commit':runtime['source_commit'],'scope':'Author ScaleBridge history/reset/observation and original export wrapper on CPU, raw original M weights; independent author-compatible float32 FK implementations; matched packets and independent action histories. TensorRT/CUDA numeric equivalence and full motion rollout remain separate.'}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));assert report['status']=='passed'
