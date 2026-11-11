"""Run the remaining upstream cases through the public differentiable API.

Each run writes an independent report, including failures. See CASES for the
upstream direction, tolerance and step; initial-contact uses a project check.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import polyfem.diff as diff
from polyfempy import polyfempy as backend
from tools.validation._boundary_config import ROOT, configuration


# filename, variable, directions, h, rtol, direction type
CASES = {
    'shape-contact': ('shape-contact-opt.json','shape',1,1e-7,7e-7,'gradient'),
    'shape-contact-adhesion': ('shape-contact-adhesion-opt.json','shape',1,1e-7,2.4e-3,'gradient'),
    'shape-contact-smooth': ('shape-contact-opt.json','shape',1,1e-6,7.5e-7,'gradient'),
    'shape-contact-force-norm': ('shape-contact-force-norm-opt.json','shape',3,1e-7,1.3e-6,'random'),
    'shape-contact-force-norm-adhesion': ('shape-contact-force-norm-opt-adhesion.json','shape',3,1e-7,5.3e-2,'random'),
    'shape-contact-force-norm-3d': ('shape-contact-force-norm-3d-opt.json','shape',3,1e-6,2.3e-7,'random'),
    'shape-pressure-nodes-3d': ('shape-pressure-nodes-3d-opt.json','shape',1,1e-7,4.5e-5,'random'),
    'shape-transient-friction-sdf': ('shape-transient-friction-sdf-opt.json','shape',3,1e-7,2.7e-6,'random'),
    'shape-transient-stress-3d-frictionless-fast': ('shape-transient-stress-3d-frictionless-fast-opt.json','shape',1,1e-5,1e-3,'gradient'),
    'shape-transient-smooth': ('shape-transient-friction-opt.json','shape',3,1e-6,4.9e-6,'random'),
    'initial-contact': ('initial-contact-opt.json','initial',3,1e-5,1.9e-6,'random'),
    'initial-contact-smooth': ('initial-contact-smooth-opt.json','initial',3,1e-6,6e-7,'random'),
    'barycenter': ('barycenter-opt.json','initial',1,1e-6,8.1e-7,'velocity'),
    'node-trajectory': ('node-trajectory-opt.json','material',3,1e-5,4.9e-5,'random'),
    'homogenize-stress': ('homogenize-stress-opt.json','shape',3,1e-7,7.5e-7,'interior'),
}


def validate(case, output):
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    filename,kind,repeat,h,rtol,direction_kind=CASES[case]
    report={'case':case,'source':filename,'passed':False,'checks':[],'rtol':rtol,'atol':0.,
            'direction_kind':direction_kind,'scope':'Python adaptation; C++ test and optimizer convergence not run',
            'tolerance_source':'project check using related initial-target tolerance; no standalone C++ test found' if case=='initial-contact' else 'polyfem/tests/test_diff.cpp',
            'adaptations':['absolute paths','one thread','SimplicialLDLT','independent output',
                           'PyTorch RNG seed 42 instead of C++ RNG','second FD step',
                           'independent scalar API calls summed in Python'],
            'commits':{n:subprocess.check_output(['git','-C',str(ROOT/n),'rev-parse','HEAD'],text=True).strip()
                       for n in ('polyfem','differentiability-data')}}
    try:
        spec=json.loads((ROOT/'differentiability-data/input'/filename).read_text())
        configs=[configuration(Path(s['path']).stem,output/f'state{i}') for i,s in enumerate(spec['states'])]
        if case=='homogenize-stress':
            original=json.loads((ROOT/'differentiability-data/input'/spec['states'][0]['path']).read_text())
            configs[0]['solver']['linear']=copy.deepcopy(original['solver']['linear'])
            report['adaptations'].remove('SimplicialLDLT')
            report['adaptations'].append('preserve original SPQR for constrained homogenization')
        if case in ('shape-contact-smooth','initial-contact-smooth','shape-transient-smooth'):
            for config in configs:
                config.setdefault('contact',{}).update(use_gcp_formulation=True,alpha_t=.95)
                if case!='shape-transient-smooth': config['contact']['use_convergent_formulation']=False
                if case!='shape-contact-smooth': config['contact']['friction_coefficient']=0
                if case=='shape-transient-smooth':config['solver']['nonlinear']['grad_norm_tol']=1e-8
            report['test_overrides']='Matched upstream smooth test contact/solver overrides before session setup'
        config=configs[0]
        session=backend.DifferentiableSession();session.set_settings(config)
        metadata=session.model_metadata()
        vertices=torch.tensor(session.mesh_vertices(),dtype=torch.float64)
        ids=torch.arange(len(vertices))
        if kind=='shape':
            selection=spec['variable_to_simulation'][0].get('active_geometry_nodes')
            if selection:
                assert selection['type'] in ('boundary','boundary_excluding_surface')
                ids=torch.tensor(session.boundary_vertex_ids(selection['selection'],selection['type']=='boundary_excluding_surface'),dtype=torch.long)
            x=vertices[ids].clone().requires_grad_()
            if isinstance(spec['parameters'],list) and spec['parameters'] and isinstance(spec['parameters'][0]['number'],int):
                assert x.numel()==spec['parameters'][0]['number']
        elif kind=='initial':
            initial=torch.tensor(session.initial_condition_parameters(),dtype=torch.float64)
            x=initial.reshape(2,-1).T.contiguous().requires_grad_()
        else:
            assert metadata['n_vertices']==metadata['n_bases']
            material=config['materials'];E,nu=material['E'],material['nu']
            assert metadata['dimension']==2
            x=torch.tensor([E*nu/(1-nu**2),E/(2*(1+nu))],dtype=torch.float64).repeat(metadata['n_elements'],1).requires_grad_()
            targets=10*torch.rand(vertices.shape,dtype=vertices.dtype,generator=torch.Generator().manual_seed(42))
            target_file=output/'node_targets.txt'
            np.savetxt(target_file,torch.cat((vertices,targets),dim=1).numpy())
            spec['functionals']=[{'type':'node-target','state':0,'target_data_path':str(target_file)}]
            report['node_targets']=targets.tolist()
            report['adaptations'].append('NodeTargetForm file constructor instead of C++ direct constructor; target assignments generated in mesh vertex order')
        del session
        objects=[]
        for functional in spec['functionals']:
            spatial=functional.get('static_objective',functional)
            if 'matching' in spatial:
                assert spatial.pop('matching')=='exact'
                report['adaptations'].append('omit obsolete matching=exact field, not read by TargetForm')
            objective=diff.objectives.create(functional['type'],**{k:v for k,v in functional.items() if k!='type'})
            if kind=='shape':objects.append(diff.shape(config,objective=objective))
            elif kind=='initial':
                target=configs[1] if spatial.get('target_state')==1 else None
                objects.append(diff.initial(config,objective=objective,target=target))
            else:objects.append(objective)
        def evaluate(obj,x):
            if kind=='shape':return obj(V=vertices.index_copy(0,ids,x))
            if kind=='initial':return obj(initial_condition=x)
            return diff.material_opt(diff.model(config),x,objective=obj)()
        def loss(x):return sum(evaluate(obj,x) for obj in objects)
        report.update(metadata=metadata,parameter_shape=list(x.shape),selected_vertices=ids.tolist() if kind=='shape' else None,
                      objectives=spec['functionals'],configs=configs)
        initial_values=[];gradients=[]
        for obj in objects:
            value=evaluate(obj,x);g,=torch.autograd.grad(value,x)
            initial_values.append(value.item());gradients.append(g)
        value=sum(initial_values);gradient=sum(gradients)
        report.update(values=initial_values,value=value,gradient=gradient.tolist())
        generator=torch.Generator().manual_seed(42)
        for trial in range(repeat):
            if direction_kind=='gradient':direction=gradient/gradient.norm()
            elif direction_kind=='velocity':
                direction=torch.zeros_like(x);direction[:,1]=torch.tensor([-2.,-1.],dtype=x.dtype).repeat(metadata['n_bases'])
            else:
                direction=torch.rand(x.shape,dtype=x.dtype,generator=generator)
                if direction_kind=='interior':
                    lower,upper=vertices.amin(0),vertices.amax(0)
                    boundary=((vertices<lower+1e-5)|(vertices>upper-1e-5)).any(1)
                    direction[boundary]=0
            actual=(gradient*direction).sum().item()
            for step in (h,3*h):
                with torch.no_grad():fd=((loss(x+step*direction)-loss(x-step*direction))/(2*step)).item()
                error=abs(actual-fd)/max(abs(actual),abs(fd),1e-30)
                check={'trial':trial,'h':step,'autograd':actual,'finite_difference':fd,'relative_error':error,'passed':error<=rtol}
                report['checks'].append(check);print(json.dumps(check),flush=True)
                report.setdefault('directions',{})[str(trial)]=direction.tolist()
                (output/'progress.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        scalar=torch.zeros((),dtype=x.dtype,requires_grad=True)
        outer,=torch.autograd.grad(loss(x.detach()+scalar*direction).square(),scalar)
        expected=2*value*(gradient*direction).sum().item()
        report['mapping']={'autograd':outer.item(),'expected':expected,'passed':abs(outer.item()-expected)<=1e-10+1e-9*abs(expected)}
        report['passed']=all(c['passed'] for c in report['checks']) and report['mapping']['passed']
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
        print(report['error'],flush=True)
    finally:
        report['elapsed_seconds']=time.perf_counter()-started
        (output/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=CASES,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    raise SystemExit(0 if validate(args.case,args.output)['passed'] else 1)
