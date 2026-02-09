"""Extract metrics from TensorBoard event files at specific timestep milestones."""
import os
import sys
import json


def read_tb_events(tb_dir):
    """Read TensorBoard event files and return metrics indexed by step."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(tb_dir)
    ea.Reload()

    metrics = {}
    for tag in ea.Tags().get('scalars', []):
        events = ea.Scalars(tag)
        for e in events:
            step = e.step
            if step not in metrics:
                metrics[step] = {}
            metrics[step][tag] = e.value

    return metrics


def get_closest_metric(metrics, target_step, tag):
    """Get the metric value closest to the target step."""
    best_step = None
    best_dist = float('inf')
    for step in metrics:
        if tag in metrics[step]:
            dist = abs(step - target_step)
            if dist < best_dist:
                best_dist = dist
                best_step = step
    if best_step is not None:
        return metrics[best_step][tag], best_step
    return None, None


def extract_run_metrics(run_dir, label):
    """Extract metrics for a run at milestone steps."""
    tb_logs_dir = os.path.join(run_dir, 'tb_logs')
    if not os.path.exists(tb_logs_dir):
        print(f"No TensorBoard logs for {label}")
        return None

    # Find the PPO_* subdirectory
    subdirs = [d for d in os.listdir(tb_logs_dir) if d.startswith('PPO')]
    if not subdirs:
        print(f"No PPO subdirectory in {tb_logs_dir}")
        return None

    tb_dir = os.path.join(tb_logs_dir, subdirs[0])
    metrics = read_tb_events(tb_dir)

    milestones = [500000, 1000000, 1500000, 2000000]
    result = {}

    for target in milestones:
        key = f"{target // 1000}K"
        result[key] = {}

        for tag_short, tag_full in [
            ('entropy', 'train/entropy_loss'),
            ('explained_variance', 'train/explained_variance'),
            ('approx_kl', 'train/approx_kl'),
            ('value_loss', 'train/value_loss'),
            ('fps', 'time/fps'),
        ]:
            val, actual_step = get_closest_metric(metrics, target, tag_full)
            if val is not None:
                # entropy_loss is negative of entropy in SB3
                if tag_short == 'entropy':
                    val = -val  # Convert entropy_loss to entropy (positive = more entropy)
                result[key][tag_short] = round(float(val), 6)
                result[key][f'{tag_short}_actual_step'] = actual_step

    print(f"\n=== {label} ===")
    for k, v in result.items():
        print(f"  {k}: {v}")

    return result


if __name__ == '__main__':
    base = os.path.dirname(__file__)

    all_metrics = {}
    for run_name, label in [
        ('run-a', 'Run A (20d exec-cost)'),
        ('run-b', 'Run B (20d no-exec-cost)'),
        ('run-c', 'Run C (199d no-exec-cost)'),
    ]:
        run_dir = os.path.join(base, run_name)
        m = extract_run_metrics(run_dir, label)
        if m:
            all_metrics[run_name] = m

    with open(os.path.join(base, 'tb_metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nTB metrics saved to tb_metrics.json")
