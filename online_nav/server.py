"""
NWM Navigation Server  (runs on the GPU workstation)

Protocol: ZMQ REQ/REP with pickle serialisation.

Incoming message (dict):
    type        : 'set_goal' | 'step'
    image       : np.ndarray (H, W, 3) uint8  — current robot observation
    goal_image  : np.ndarray (H, W, 3) uint8  — only required with type='set_goal'

Outgoing reply (dict):
    status      : 'ok' | 'error'
    step_dx     : float  meters  (forward,  averaged over planned trajectory)
    step_dy     : float  meters  (lateral,  averaged over planned trajectory)
    total_dyaw  : float  radians (total yaw change)
    debug_img   : np.ndarray (224, 224, 3) uint8  (NWM final-frame prediction)
    message     : str  (present only when status='error')

Usage:
    python server.py --ckp logs/nwm_cdit_xl/checkpoints/0100000.pth.tar \
                     --port 5555 \
                     [--num_samples 30] [--diffusion_steps 50] \
                     [--metric_waypoint_spacing 0.05]
"""

import argparse
import pickle
import traceback

import numpy as np
import zmq

from nwm_planner import NWMPlanner


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckp',   required=True, help='path to .pth.tar checkpoint')
    p.add_argument('--port',  type=int, default=5555)
    p.add_argument('--device', default='cuda')
    # planner hyperparams
    p.add_argument('--num_samples',       type=int,   default=30)
    p.add_argument('--topk',              type=int,   default=5)
    p.add_argument('--opt_steps',         type=int,   default=1)
    p.add_argument('--rollout_stride',    type=int,   default=4)
    p.add_argument('--len_traj_pred',     type=int,   default=8)
    p.add_argument('--diffusion_steps',   type=int,   default=50)
    p.add_argument('--num_repeat_eval',   type=int,   default=1)
    # robot params (must match data_config.yaml for your robot)
    p.add_argument('--metric_waypoint_spacing', type=float, default=0.05)
    p.add_argument('--mu_x',    type=float, default=-0.1,  help='CEM initial mean dx (normalised)')
    p.add_argument('--sigma_x', type=float, default=0.05,  help='CEM initial std dx (normalised)')
    return p.parse_args()


def main():
    args = parse_args()

    planner = NWMPlanner(
        ckp_path=args.ckp,
        device=args.device,
        num_samples=args.num_samples,
        topk=args.topk,
        opt_steps=args.opt_steps,
        rollout_stride=args.rollout_stride,
        len_traj_pred=args.len_traj_pred,
        diffusion_steps=args.diffusion_steps,
        num_repeat_eval=args.num_repeat_eval,
        mu_init=[args.mu_x, 0.0, 0.0],
        sigma_init=[args.sigma_x, 0.1, 0.1],
        metric_waypoint_spacing=args.metric_waypoint_spacing,
    )

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f'tcp://*:{args.port}')
    print(f"NWM server listening on port {args.port}")

    while True:
        raw = sock.recv()
        try:
            msg = pickle.loads(raw)
            img = msg['image']       # (H, W, 3) uint8
            mtype = msg.get('type', 'step')

            if mtype == 'set_goal':
                planner.set_goal(msg['goal_image'])
                planner.push_frame(img)
                reply = {'status': 'ok', 'step_dx': 0.0, 'step_dy': 0.0,
                         'total_dyaw': 0.0, 'debug_img': None}

            elif mtype == 'step':
                planner.push_frame(img)
                step_actions, total_dyaw, debug_img = planner.plan()
                # Return the first step's (dx, dy) so the robot executes one step then replans.
                # step_actions shape: (len_traj_pred, 2), but all rows are identical (straight-line)
                reply = {
                    'status':      'ok',
                    'step_dx':     float(step_actions[0, 0]),
                    'step_dy':     float(step_actions[0, 1]),
                    'total_dyaw':  total_dyaw,
                    'debug_img':   debug_img,
                }
            else:
                reply = {'status': 'error', 'message': f'Unknown type: {mtype}'}

        except Exception:
            reply = {'status': 'error', 'message': traceback.format_exc()}
            print(reply['message'])

        sock.send(pickle.dumps(reply))


if __name__ == '__main__':
    main()
