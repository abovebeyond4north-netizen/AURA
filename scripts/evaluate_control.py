"""Evaluate a saved neural actor against a random-policy development comparator."""
import argparse,json
from pathlib import Path
import torch
from aura.config import Config, seed_everything
from aura.planning.actor import Actor
from aura.world_model.dynamics import WorldModel
from aura.evaluation.control import evaluate_control


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--actor',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--episodes',type=int,default=5)
    parser.add_argument('--max-steps',type=int,default=40)
    parser.add_argument('--seed-index',type=int,default=0)
    args=parser.parse_args()
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    config=Config.model_validate(payload['metadata']['config'])
    seed_everything(config.seed)
    world=WorldModel(config.latent_dim,config.hidden_dim,config.ensemble_size,config.lr,'cpu')
    world.restore(args.checkpoint); world.eval()
    actor=Actor(config.latent_dim,config.lr,'cpu')
    actor_path=args.actor or args.checkpoint.with_name('actor.pt')
    actor.restore(actor_path);actor.eval()
    report=evaluate_control(world,actor,episodes=args.episodes,max_steps=args.max_steps,seed_index=args.seed_index)
    report['actor_train_steps']=actor.train_steps
    report['world_train_steps']=world.train_steps
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report['summary'],indent=2,allow_nan=False))

if __name__=='__main__':main()
