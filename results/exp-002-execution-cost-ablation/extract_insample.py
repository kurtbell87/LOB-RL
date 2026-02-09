"""Extract in-sample (rollout) reward and episode length from TensorBoard."""
import os
import sys
import json


def read_tb_events(tb_dir):
    """Read TensorBoard event files and return all scalar events."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(tb_dir)
    ea.Reload()

    results = {}
    for tag in ea.Tags().get('scalars', []):
        events = ea.Scalars(tag)
        results[tag] = [(e.step, e.value) for e in events]

    return results


def extract_insample(run_dir, label):
    """Extract final in-sample rollout reward."""
    tb_logs_dir = os.path.join(run_dir, 'tb_logs')
    subdirs = [d for d in os.listdir(tb_logs_dir) if d.startswith('PPO')]
    tb_dir = os.path.join(tb_logs_dir, subdirs[0])

    data = read_tb_events(tb_dir)

    print(f"\n=== {label} ===")
    print(f"Available tags: {list(data.keys())}")

    # Get rollout/ep_rew_mean (in-sample return) - last N values
    if 'rollout/ep_rew_mean' in data:
        rew_data = data['rollout/ep_rew_mean']
        last_10 = rew_data[-10:]
        last_val = rew_data[-1]
        print(f"  ep_rew_mean (last): step={last_val[0]}, value={last_val[1]:.4f}")
        print(f"  ep_rew_mean (last 10 avg): {sum(v for _, v in last_10)/len(last_10):.4f}")

    if 'rollout/ep_len_mean' in data:
        len_data = data['rollout/ep_len_mean']
        last_val = len_data[-1]
        print(f"  ep_len_mean (last): step={last_val[0]}, value={last_val[1]:.4f}")

    return data


if __name__ == '__main__':
    base = os.path.dirname(__file__)

    results = {}
    for run_name, label in [
        ('run-a', 'Run A (20d exec-cost)'),
        ('run-b', 'Run B (20d no-exec-cost)'),
        ('run-c', 'Run C (199d no-exec-cost)'),
    ]:
        run_dir = os.path.join(base, run_name)
        data = extract_insample(run_dir, label)

        result = {}
        if 'rollout/ep_rew_mean' in data:
            vals = data['rollout/ep_rew_mean']
            result['final_ep_rew_mean'] = vals[-1][1]
            result['final_ep_rew_mean_step'] = vals[-1][0]
        if 'rollout/ep_len_mean' in data:
            vals = data['rollout/ep_len_mean']
            result['final_ep_len_mean'] = vals[-1][1]
        results[run_name] = result

    with open(os.path.join(base, 'insample_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nIn-sample metrics saved to insample_metrics.json")
