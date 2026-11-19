import json
from pathlib import Path
import torch,numpy as np
from safetensors.torch import save_file
from native_runtime import load_policy
from tensor_policy import load
from motion_challenge.validation import sha256_file

def prepare(root,runtime,source,output,artifacts,device):
    root=Path(root);source=Path(source);output=Path(output);paths={k:Path(v) for k,v in artifacts.items()}
    if (output/'conversion.json').exists() and (output/'parity.json').exists():
        receipt=json.loads((output/'conversion.json').read_text())
        if receipt.get('source_sha256')==sha256_file(source) and receipt.get('batch_wrapper_sha256')==sha256_file(root/'tensor_native.py') and all((output/n).exists() and sha256_file(output/n)==d for n,d in receipt.get('files',{}).items()):
            return output
    assert sha256_file(source)==runtime['weights_sha256']
    for k,p in paths.items():assert sha256_file(p)==runtime['artifacts'][k]['sha256']
    policy,meta,native=load_policy(paths,source)
    out=output;out.mkdir(parents=True,exist_ok=True)
    save_file({k:v.contiguous().clone() for k,v in policy.state_dict().items()},str(out/'policy.safetensors'))
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    receipt={'source_sha256':sha256_file(source),'artifacts_sha256':{k:sha256_file(p) for k,p in paths.items()},'files':{n:sha256_file(out/n) for n in ['policy.safetensors','metadata.json']},'batch_wrapper_sha256':sha256_file(root/'tensor_native.py')}
    (out/'conversion.json').write_text(json.dumps(receipt,indent=2)+'\n')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(2026)
    rows=[];maximum=0.;gpu,_=load(out,device,runtime['weights_sha256']);H=meta['history_buffer_size'];F=len(meta['future_idx']);B=len(meta['selected_body_names'])
    for size in (1,7,40,256):
     q=torch.randn(size,H,4);q/=q.norm(dim=-1,keepdim=True)
     tq=torch.randn(size,F,B,4);tq/=tq.norm(dim=-1,keepdim=True)
     inputs=[q,torch.randn(size,H,3)*.1,torch.randn(size,H,29)*.1,torch.randn(size,H,29)*.1,torch.randn(size,H,29)*.1,torch.randn(size,F,B,3)*.1,tq,torch.full((size,),7),torch.tensor(meta['future_idx'])[None,:,None].expand(size,-1,-1)]
     with torch.inference_mode():
      old=[policy(*(x[i:i+1] for x in inputs)) for i in range(size)]
      expected=[torch.cat([x[j] for x in old]) for j in range(2)]
      actual=gpu(*(x.to(device) for x in inputs))
     error=max(float((a.cpu()-e).abs().max()) for a,e in zip(actual,expected));maximum=max(maximum,error)
     for a,e in zip(actual,expected):np.testing.assert_allclose(a.cpu().numpy(),e.numpy(),rtol=2e-5,atol=2e-5)
     rows.append({'batch':size,'max_abs_error':error});print(rows[-1],flush=True)
    report={'status':'passed','source_sha256':runtime['weights_sha256'],'max_abs_error':maximum,'cases':rows,'reference':'Unmodified pinned author scalar CPU wrapper','files':receipt['files']}
    (out/'parity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    
    return output
