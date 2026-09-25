import argparse
import math
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import random
from exogenous_generator import ExogenousGenerator
from pretrain_exogenous_generator import load_u
from mlp import build_mlp
from scm import forward_scm
import csv
from datetime import datetime
import copy

def nominal_kl(generator, nominal_u):
    """Estimate KL(P_nominal || Q_generator) with gradients."""
    generator.clear_cache()
    log_p = -0.5 * nominal_u.square() - 0.5 * math.log(2.0 * math.pi)
    log_q = generator.log_prob(nominal_u)
    if log_q.shape != nominal_u.shape:
        raise ValueError('Unexpected log-density shape.')
    log_ratio = log_p - log_q
    if not torch.isfinite(log_ratio).all().item():
        raise ValueError('Non-finite log-density ratio.')
    return log_ratio.mean()

def generate_batch(generators, noise):
    """Generate an SCM batch from the current generators."""
    coordinates = ['U1', 'U2', 'U3', 'U4']
    generated_u = []
    for i, coordinate in enumerate(coordinates):
        generator = generators[coordinate]
        generator.clear_cache()
        u = generator(noise[:, i])
        generated_u.append(u)
    uy = noise[:, 4]
    x, y = forward_scm(*generated_u, uy)
    return (x, y)

def update_generator(generators, model, selected_coordinate, nominal_u, noise, optimizer, multiplier, kl_budget, dual_lr, train_mean, train_std):
    """Update one generator and its multiplier."""
    coordinates = ['U1', 'U2', 'U3', 'U4']
    batch_size = noise.shape[0]
    for coordinate, generator in generators.items():
        generator.requires_grad_(coordinate == selected_coordinate)
        generator.zero_grad(set_to_none=True)
        generator.clear_cache()
    model.eval()
    model.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    active_generator = generators[selected_coordinate]
    pass
    x, y = generate_batch(generators, noise)
    prediction = model(normalize_inputs(x, train_mean, train_std))
    if x.shape != (batch_size, 4):
        raise ValueError('Unexpected feature shape.')
    if y.shape != (batch_size, 1):
        raise ValueError('Unexpected target shape.')
    if prediction.shape != y.shape:
        raise ValueError('Prediction and target shapes differ.')
    mse = F.mse_loss(prediction, y)
    if not torch.isfinite(mse).item():
        raise ValueError('Prediction MSE is not finite.')
    if not mse.requires_grad:
        raise ValueError('Prediction MSE has no gradient graph.')
    pass
    pass
    pass
    pass
    pass
    'Compute a differentiable KL estimate.'
    kl = nominal_kl(active_generator, nominal_u)
    constraint_residual = kl - kl_budget
    'Minimize this loss to increase prediction MSE while controlling KL.'
    generator_loss = -mse + multiplier * constraint_residual
    if not torch.isfinite(generator_loss).item():
        raise ValueError('Generator loss is not finite.')
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    generator_states_before = {coordinate: {name: value.detach().clone() for name, value in generator.state_dict().items()} for coordinate, generator in generators.items()}
    mlp_state_before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    optimizer.zero_grad(set_to_none=True)
    generator_loss.backward()
    gradient_squared_sum = 0.0
    for name, parameter in active_generator.named_parameters():
        if parameter.grad is None:
            raise ValueError(f'Missing gradient: {name}')
        if not torch.isfinite(parameter.grad).all().item():
            raise ValueError(f'Non-finite gradient: {name}')
        gradient_squared_sum += parameter.grad.double().square().sum().item()
    gradient_norm = math.sqrt(gradient_squared_sum)
    if gradient_norm == 0.0:
        raise ValueError('The selected generator has zero gradient.')
    for coordinate, generator in generators.items():
        if coordinate != selected_coordinate:
            if any((p.grad is not None for p in generator.parameters())):
                raise ValueError(f'Frozen generator received gradients: {coordinate}')
    if any((p.grad is not None for p in model.parameters())):
        raise ValueError('The frozen MLP received parameter gradients.')
    optimizer.step()
    for generator in generators.values():
        generator.clear_cache()
    for coordinate, generator in generators.items():
        state_after = generator.state_dict()
        if not all((torch.isfinite(value).all().item() for value in state_after.values())):
            raise ValueError(f'Non-finite model state: {coordinate}')
        changed = any((not torch.equal(value, generator_states_before[coordinate][name]) for name, value in state_after.items()))
        if changed != (coordinate == selected_coordinate):
            raise ValueError(f'Unexpected parameter change: {coordinate}')
        pass
    mlp_changed = any((not torch.equal(value, mlp_state_before[name]) for name, value in model.state_dict().items()))
    if mlp_changed:
        raise ValueError('The frozen MLP changed.')
    pass
    pass
    pass
    pass
    with torch.no_grad():
        x_after, y_after = generate_batch(generators, noise)
        prediction_after = model(normalize_inputs(x_after, train_mean, train_std))
        mse_after = F.mse_loss(prediction_after, y_after)
        kl_after = nominal_kl(active_generator, nominal_u)
    for generator in generators.values():
        generator.clear_cache()
    if not torch.isfinite(mse_after).item():
        raise ValueError('Post-update MSE is not finite.')
    if not torch.isfinite(kl_after).item():
        raise ValueError('Post-update KL is not finite.')
    residual_after = kl_after.item() - kl_budget
    multiplier_before = multiplier
    multiplier = max(0.0, multiplier_before + dual_lr * residual_after)
    if not math.isfinite(multiplier):
        raise ValueError('Updated multiplier is not finite.')
    pass
    pass
    pass
    pass
    metrics = {'kl': kl_after.item(), 'kl_residual': residual_after}
    return (multiplier, metrics)

def update_mlp(generators, model, noise, optimizer, train_mean, train_std):
    """Update the MLP once while keeping all generators fixed."""
    for generator in generators.values():
        generator.requires_grad_(False)
        generator.zero_grad(set_to_none=True)
        generator.clear_cache()
    generator_states_before = {coordinate: {name: value.detach().clone() for name, value in generator.state_dict().items()} for coordinate, generator in generators.items()}
    mlp_state_before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        x, y = generate_batch(generators, noise)
    for generator in generators.values():
        generator.clear_cache()
    model.train()
    model.requires_grad_(True)
    optimizer.zero_grad(set_to_none=True)
    prediction = model(normalize_inputs(x, train_mean, train_std))
    if prediction.shape != y.shape:
        raise ValueError('Prediction and target shapes do not match.')
    loss = F.mse_loss(prediction, y)
    if not torch.isfinite(loss).item():
        raise ValueError('MLP loss is not finite.')
    loss.backward()
    gradient_squared_sum = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise ValueError(f'Missing MLP gradient: {name}')
        if not torch.isfinite(parameter.grad).all().item():
            raise ValueError(f'Non-finite MLP gradient: {name}')
        gradient_squared_sum += parameter.grad.double().square().sum().item()
    gradient_norm = math.sqrt(gradient_squared_sum)
    if gradient_norm == 0.0:
        raise ValueError('MLP gradient norm is zero.')
    optimizer.step()
    for coordinate, generator in generators.items():
        changed = any((not torch.equal(value, generator_states_before[coordinate][name]) for name, value in generator.state_dict().items()))
        if changed:
            raise ValueError(f'The frozen {coordinate} changed.')
        if any((parameter.grad is not None for parameter in generator.parameters())):
            raise ValueError(f'The frozen {coordinate} received gradients.')
        pass
    for name, value in model.state_dict().items():
        if not torch.isfinite(value).all().item():
            raise ValueError(f'Non-finite MLP state: {name}')
    mlp_changed = any((not torch.equal(value, mlp_state_before[name]) for name, value in model.state_dict().items()))
    if not mlp_changed:
        raise ValueError('The MLP did not change.')
    model.eval()
    with torch.no_grad():
        prediction_after = model(normalize_inputs(x, train_mean, train_std))
        loss_after = F.mse_loss(prediction_after, y)
    if not torch.isfinite(loss_after).item():
        raise ValueError('Post-update MLP loss is not finite.')
    pass
    pass
    pass
    pass
    pass
    return (loss.item(), loss_after.item())

def load_xy(path):
    """Load fixed features and targets from a CSV file."""
    data = np.genfromtxt(path, delimiter=',', names=True, dtype=np.float64)
    data = np.atleast_1d(data)
    required_columns = ['X1', 'X2', 'X3', 'X4', 'Y']
    column_names = data.dtype.names or ()
    if any((name not in column_names for name in required_columns)):
        raise ValueError('CSV must contain X1, X2, X3, X4, and Y.')
    if data.size == 0:
        raise ValueError('The evaluation dataset is empty.')
    x_array = np.column_stack([data[name] for name in required_columns[:4]])
    y_array = data['Y'].reshape(-1, 1)
    x = torch.tensor(x_array, dtype=torch.float32)
    y = torch.tensor(y_array, dtype=torch.float32)
    if not torch.isfinite(x).all().item():
        raise ValueError('Non-finite evaluation features.')
    if not torch.isfinite(y).all().item():
        raise ValueError('Non-finite evaluation targets.')
    return (x, y)

def compute_train_normalization(train_path):
    x_train, _ = load_xy(train_path)
    if x_train.shape[0] < 2:
        raise ValueError('At least two training samples are required.')
    train_mean = x_train.mean(dim=0)
    train_std = x_train.std(dim=0, unbiased=False)
    if not torch.isfinite(train_mean).all().item():
        raise ValueError('Training input means are not finite.')
    if not torch.isfinite(train_std).all().item():
        raise ValueError('Training input standard deviations are not finite.')
    if (train_std <= 0).any().item():
        raise ValueError('Training input standard deviations must be positive.')
    return (train_mean, train_std)

def normalize_inputs(x, train_mean, train_std):
    """Normalize inputs using fixed training statistics."""
    return (x - train_mean) / train_std

def evaluate_mse(model, x, y, train_mean, train_std):
    """Evaluate MSE without updating the predictor."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            prediction = model(normalize_inputs(x, train_mean, train_std))
            if prediction.shape != y.shape:
                raise ValueError('Unexpected evaluation prediction shape.')
            mse = F.mse_loss(prediction, y)
            if not torch.isfinite(mse).item():
                raise ValueError('Evaluation MSE is not finite.')
            return mse.item()
    finally:
        model.train(was_training)

def load_ood_validation(root, model):
    """Load the 15 fixed OOD validation environments."""
    environments = {}
    parameter = next(model.parameters())
    expected_names = {f'env_{i:02d}.csv' for i in range(5)}
    for strength in ['mild', 'moderate', 'strong']:
        folder = root / strength
        paths = sorted(folder.glob('env_*.csv'))
        if {path.name for path in paths} != expected_names:
            raise ValueError(f'Expected env_00.csv through env_04.csv in {folder}')
        for path in paths:
            x, y = load_xy(path)
            x = x.to(device=parameter.device, dtype=parameter.dtype)
            y = y.to(device=parameter.device, dtype=parameter.dtype)
            environment_id = f'{strength}/{path.stem}'
            environments[environment_id] = (strength, x, y)
    return environments

def evaluate_ood(model, environments, train_mean, train_std):
    """Evaluate per-environment and aggregated OOD MSE."""
    metrics = {}
    grouped_mse = {'mild': [], 'moderate': [], 'strong': []}
    for environment_id, (strength, x, y) in environments.items():
        mse = evaluate_mse(model, x, y, train_mean, train_std)
        column_id = environment_id.replace('/', '_')
        metrics[f'ood_val_{column_id}_mse'] = mse
        grouped_mse[strength].append(mse)
    all_mse = []
    for strength, values in grouped_mse.items():
        if not values:
            raise ValueError(f'No OOD environments for {strength}.')
        metrics[f'ood_val_{strength}_mean_mse'] = math.fsum(values) / len(values)
        all_mse.extend(values)
    metrics['ood_val_mean_mse'] = math.fsum(all_mse) / len(all_mse)
    metrics['ood_val_worst_mse'] = max(all_mse)
    return metrics

def save_checkpoint(path, checkpoint):
    """Save to a temporary file before replacing the checkpoint."""
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)

def set_seed(seed):
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coordinate', choices=['U1', 'U2', 'U3', 'U4'], default='U1')
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--kl-budget', type=float, default=0.3)
    parser.add_argument('--mu-init', type=float, default=0.0, help='Initial nonnegative dual multiplier for each selected coordinate.')
    parser.add_argument('--generator-lr', type=float, default=0.0001)
    parser.add_argument('--dual-lr', type=float, default=0.01)
    parser.add_argument('--sweep', action='store_true', help='Update U1 through U4 sequentially; each coordinate receives --inner-steps adversarial updates before moving to the next.')
    parser.add_argument('--mlp-lr', type=float, default=0.001, help='Learning rate for the MLP.')
    parser.add_argument('--steps', type=int, default=1, help='Number of outer alternating training iterations.')
    parser.add_argument('--inner-steps', type=int, default=10, help='Number of adversarial updates K applied to each selected exogenous coordinate before moving to the next coordinate.')
    parser.add_argument('--validation-id', type=Path, required=True, help='Path to the fixed ID validation CSV.')
    parser.add_argument('--log-dir', type=Path, default=Path('logs'), help='Directory for training CSV logs.')
    parser.add_argument('--validation-ood', type=Path, required=True, help='Directory containing OOD validation environments.')
    parser.add_argument('--checkpoint-dir', type=Path, default=Path('checkpoints') / 'ours', help='Directory for alternating-training checkpoints.')
    parser.add_argument('--selection-metric', choices=['mean', 'worst'], required=True, help='OOD validation metric used to select the best checkpoint.')
    parser.add_argument('--seed', type=int, default=0, help='Training seed for initialization and sampling.')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='Device used for model training and evaluation.')
    parser.add_argument('--resume', type=Path, default=None, help='Path to the last checkpoint used to resume training.')
    parser.add_argument('--erm-checkpoint-dir', type=Path, default=Path('checkpoints') / 'final_oodval', help='Directory containing the seed-specific ERM checkpoints.')
    parser.add_argument('--generator-checkpoint-dir', type=Path, default=Path(__file__).resolve().parent / 'checkpoints', help='Directory containing pretrained nominal generators.')
    args = parser.parse_args()
    if not 0 <= args.seed < 2 ** 32:
        parser.error('--seed must be between 0 and 2**32 - 1.')
    if args.device == 'cuda' and (not torch.cuda.is_available()):
        parser.error('--device cuda requires an available CUDA GPU.')
    device = torch.device(args.device)
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(device)}')
    set_seed(args.seed)
    print(f'Training seed: {args.seed}')
    if not math.isfinite(args.kl_budget) or args.kl_budget < 0:
        parser.error('--kl-budget must be finite and nonnegative.')
    if not math.isfinite(args.mu_init) or args.mu_init < 0:
        parser.error('--mu-init must be finite and nonnegative.')
    if not math.isfinite(args.generator_lr) or args.generator_lr <= 0:
        parser.error('--generator-lr must be finite and positive.')
    checkpoint_dir = args.generator_checkpoint_dir
    if not math.isfinite(args.dual_lr) or args.dual_lr < 0:
        parser.error('--dual-lr must be finite and nonnegative.')
    if not math.isfinite(args.mlp_lr) or args.mlp_lr <= 0:
        parser.error('--mlp-lr must be finite and positive.')
    if args.steps < 1:
        parser.error('--steps must be at least 1.')
    if args.inner_steps < 1:
        parser.error('--inner-steps must be at least 1.')
    coordinates = ['U1', 'U2', 'U3', 'U4']
    generators = {}
    for coordinate in coordinates:
        checkpoint_path = checkpoint_dir / f'nominal_{coordinate.lower()}.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if checkpoint['coordinate'] != coordinate:
            raise ValueError(f"Expected {coordinate}, got {checkpoint['coordinate']}.")
        generator = ExogenousGenerator(num_bins=checkpoint['num_bins'], bound=checkpoint['bound'])
        generator.load_state_dict(checkpoint['state_dict'])
        generator = generator.to(device)
        generator.clear_cache()
        generator.requires_grad_(False)
        generators[coordinate] = generator
        print(f'{coordinate}: loaded (bound={generator.bound}, bins={generator.num_bins}, path={checkpoint_path.resolve()})')
    torch.manual_seed(args.seed)
    model = build_mlp().to(device)
    erm_path = args.erm_checkpoint_dir / f'erm_seed{args.seed}_best.pt'
    erm_checkpoint = torch.load(erm_path, map_location='cpu', weights_only=True)
    if erm_checkpoint['seed'] != args.seed:
        raise ValueError('ERM checkpoint seed mismatch.')
    if erm_checkpoint['selection_metric'] != 'worst_ood_validation_mse':
        raise ValueError('Unexpected ERM selection metric.')
    if erm_checkpoint['normalization']['source'] != 'train_only':
        raise ValueError('ERM normalization must use training data only.')
    model.load_state_dict(erm_checkpoint['model_state_dict'], strict=True)
    import hashlib
    erm_sha256 = hashlib.sha256(erm_path.read_bytes()).hexdigest()
    print(f'ERM checkpoint: {erm_path.resolve()}')
    print(f"ERM best epoch: {erm_checkpoint['best_epoch']}")
    current_mean, current_std = compute_train_normalization(args.train)
    normalization = erm_checkpoint['normalization']
    train_mean = normalization['x_mean'].reshape(-1).clone()
    train_std = normalization['x_std'].reshape(-1).clone()
    if train_mean.shape != (4,) or train_std.shape != (4,):
        raise ValueError('Unexpected ERM normalization shape.')
    if not torch.isfinite(train_mean).all().item():
        raise ValueError('Non-finite ERM input mean.')
    if not torch.isfinite(train_std).all().item() or (train_std <= 0).any().item():
        raise ValueError('Invalid ERM input standard deviation.')
    if not torch.allclose(current_mean, train_mean, rtol=1e-05, atol=1e-07):
        raise ValueError('Training mean differs from the ERM checkpoint.')
    if not torch.allclose(current_std, train_std, rtol=1e-05, atol=1e-07):
        raise ValueError('Training std differs from the ERM checkpoint.')
    model_parameter = next(model.parameters())
    train_mean = train_mean.to(device=model_parameter.device, dtype=model_parameter.dtype)
    train_std = train_std.to(device=model_parameter.device, dtype=model_parameter.dtype)
    print('Training input mean:', train_mean.cpu().tolist())
    print('Training input std:', train_std.cpu().tolist())
    model.eval()
    model.requires_grad_(False)
    mlp_optimizer = torch.optim.Adam(model.parameters(), lr=args.mlp_lr)
    'Development batch size.'
    batch_size = 128
    sampling_seed = args.seed + 12345
    rng = torch.Generator(device='cpu')
    rng.manual_seed(sampling_seed)
    print(f'Sampling seed: {sampling_seed}')
    selected_coordinates = coordinates if args.sweep else [args.coordinate]
    nominal_samples = {coordinate: load_u(args.train, coordinate).to(device=device, dtype=torch.float32) for coordinate in selected_coordinates}
    optimizers = {coordinate: torch.optim.Adam(generators[coordinate].parameters(), lr=args.generator_lr) for coordinate in selected_coordinates}
    multipliers = {coordinate: args.mu_init for coordinate in selected_coordinates}
    start_step = 1
    resume_checkpoint = None
    previous_best = None
    if args.resume is not None:
        resume_checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        if resume_checkpoint['method'] != 'ours':
            raise ValueError('Expected an Ours checkpoint.')
        saved_config = resume_checkpoint['config']
        if saved_config.get('predictor_initialization') != 'erm':
            raise ValueError('Resume checkpoint was not initialized from ERM.')
        if saved_config.get('erm_checkpoint_sha256') != erm_sha256:
            raise ValueError('Resume ERM checkpoint mismatch.')
        for name in ['seed', 'device', 'kl_budget', 'generator_lr', 'dual_lr', 'mlp_lr', 'selection_metric']:
            if saved_config[name] != getattr(args, name):
                raise ValueError(f'Resume configuration mismatch: {name}')
        saved_inner_steps = saved_config.get('inner_steps', 1)
        if saved_inner_steps != args.inner_steps:
            raise ValueError(f'Resume configuration mismatch: inner_steps (saved={saved_inner_steps}, requested={args.inner_steps})')
        if saved_config['selected_coordinates'] != selected_coordinates:
            raise ValueError('Resume coordinate selection mismatch.')
        if saved_config['batch_size'] != batch_size:
            raise ValueError('Resume batch size mismatch.')
        for name in ['train', 'validation_id', 'validation_ood']:
            if Path(saved_config[name]).resolve() != getattr(args, name).resolve():
                raise ValueError(f'Resume dataset path mismatch: {name}')
        start_step = int(resume_checkpoint['step']) + 1
        if args.steps < start_step:
            raise ValueError('--steps must exceed the saved checkpoint step.')
        for coordinate, generator in generators.items():
            expected_config = {'num_bins': generator.num_bins, 'bound': generator.bound}
            if resume_checkpoint['generator_configs'][coordinate] != expected_config:
                raise ValueError(f'Resume generator configuration mismatch: {coordinate}')
            generator.load_state_dict(resume_checkpoint['generator_state_dicts'][coordinate])
            generator.clear_cache()
        model.load_state_dict(resume_checkpoint['mlp_state_dict'])
        mlp_optimizer.load_state_dict(resume_checkpoint['mlp_optimizer_state_dict'])
        for coordinate, optimizer in optimizers.items():
            optimizer.load_state_dict(resume_checkpoint['generator_optimizer_state_dicts'][coordinate])
        multipliers = dict(resume_checkpoint['multipliers'])
        normalization = resume_checkpoint['input_normalization']
        if normalization['feature_names'] != ['X1', 'X2', 'X3', 'X4']:
            raise ValueError('Resume feature order mismatch.')
        if normalization['std_unbiased'] is not False:
            raise ValueError('Resume standard deviation convention mismatch.')
        for current, saved_stat in [(train_mean, normalization['mean']), (train_std, normalization['std'])]:
            if not torch.equal(current.detach().cpu(), saved_stat):
                raise ValueError('Resume normalization mismatch.')
        train_mean = normalization['mean'].to(device=device, dtype=model_parameter.dtype)
        train_std = normalization['std'].to(device=device, dtype=model_parameter.dtype)
        previous_best = torch.load(args.resume.parent / 'best.pt', map_location='cpu', weights_only=True)
        selection = resume_checkpoint['selection']
        if previous_best['step'] != selection['best_step']:
            raise ValueError('Previous best checkpoint step mismatch.')
        if previous_best['selection']['metric'] != selection['metric']:
            raise ValueError('Previous best selection metric mismatch.')
        if previous_best['metrics'][selection['metric']] != selection['best_score']:
            raise ValueError('Previous best checkpoint score mismatch.')
        if resume_checkpoint['sampling_rng_device'] != str(rng.device):
            raise ValueError('Sampling RNG device mismatch.')
        rng.set_state(resume_checkpoint['sampling_rng_state'])
        torch.set_rng_state(resume_checkpoint['torch_rng_state'])
        if device.type == 'cuda':
            cuda_states = resume_checkpoint['cuda_rng_states']
            if torch.cuda.device_count() != 1 or not cuda_states:
                raise ValueError('Exactly one visible GPU and saved CUDA states required.')
            torch.cuda.set_rng_state(cuda_states[0], device=0)
            print(f'[RESUME] Saved CUDA 0 of {len(cuda_states)} -> current CUDA 0')
        print(f"Resuming from step {resume_checkpoint['step']}; next step: {start_step}")
    validation_x, validation_y = load_xy(args.validation_id)
    model_parameter = next(model.parameters())
    validation_x = validation_x.to(device=model_parameter.device, dtype=model_parameter.dtype)
    validation_y = validation_y.to(device=model_parameter.device, dtype=model_parameter.dtype)
    initial_validation_mse = evaluate_mse(model, validation_x, validation_y, train_mean, train_std)
    print(f'\nID validation samples: {validation_x.shape[0]}')
    print(f'Initial ID validation MSE: {initial_validation_mse:.8f}')
    ood_environments = load_ood_validation(args.validation_ood, model)
    initial_ood_metrics = evaluate_ood(model, ood_environments, train_mean, train_std)
    if args.resume is None:
        score_pairs = [('ood_val_worst_mse', 'best_ood_val_worst_mse'), ('ood_val_mean_mse', 'best_ood_val_average_mse')]
        for current_key, saved_key in score_pairs:
            actual = initial_ood_metrics[current_key]
            expected = float(erm_checkpoint[saved_key])
            print(f'ERM {current_key}: saved={expected:.8f}, restored={actual:.8f}')
            if not math.isclose(actual, expected, rel_tol=1e-05, abs_tol=1e-07):
                raise ValueError(f'ERM validation score mismatch: {current_key}')
        print('[PASS] ERM validation scores reproduced.')
    print(f'OOD validation environments: {len(ood_environments)}')
    print(f"Initial OOD validation mean MSE: {initial_ood_metrics['ood_val_mean_mse']:.8f}")
    print(f"Initial OOD validation worst MSE: {initial_ood_metrics['ood_val_worst_mse']:.8f}")
    history = []
    args.log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    log_path = args.log_dir / f'training_{timestamp}.csv'
    run_checkpoint_dir = args.checkpoint_dir / timestamp
    run_checkpoint_dir.mkdir(parents=True, exist_ok=False)
    last_checkpoint_path = run_checkpoint_dir / 'last.pt'
    best_checkpoint_path = run_checkpoint_dir / 'best.pt'
    selection_key = {'mean': 'ood_val_mean_mse', 'worst': 'ood_val_worst_mse'}[args.selection_metric]
    best_score = math.inf
    best_step = None
    if resume_checkpoint is not None:
        selection = resume_checkpoint['selection']
        if selection['metric'] != selection_key:
            raise ValueError('Resume selection metric mismatch.')
        best_score = selection['best_score']
        best_step = selection['best_step']
        save_checkpoint(best_checkpoint_path, previous_best)
        print(f'Preserved best checkpoint: step {best_step}, score={best_score:.8f}')
    run_config = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
    run_config.update({'batch_size': batch_size, 'selected_coordinates': list(selected_coordinates), 'kl_direction': 'P_nominal || Q_generator', 'kl_budget_scope': 'per_coordinate', 'sampling_seed': sampling_seed, 'predictor_initialization': 'erm', 'erm_checkpoint_path': str(erm_path.resolve()), 'erm_checkpoint_sha256': erm_sha256, 'erm_best_epoch': erm_checkpoint['best_epoch'], 'erm_initial_ood_val_worst_mse': erm_checkpoint['best_ood_val_worst_mse'], 'mlp_weight_decay': 0.0})
    print(f'Checkpoint directory: {run_checkpoint_dir.resolve()}')
    fieldnames = ['step', 'mlp_mse_before', 'mlp_mse_after', 'id_validation_mse']
    fieldnames.extend(initial_ood_metrics.keys())
    for coordinate in selected_coordinates:
        fieldnames.extend([f'{coordinate}_kl', f'{coordinate}_kl_residual', f'{coordinate}_multiplier'])
    with log_path.open('x', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'step': start_step - 1, 'id_validation_mse': initial_validation_mse, **initial_ood_metrics})
    print(f'Training log: {log_path.resolve()}')
    if resume_checkpoint is None:
        best_score = float(initial_ood_metrics[selection_key])
        best_step = 0
        if not math.isfinite(best_score):
            raise ValueError('Initial validation score is not finite.')
        initial_checkpoint = {'format_version': 1, 'method': 'ours', 'step': 0, 'config': run_config, 'log_path': str(log_path.resolve()), 'mlp_architecture': {'input_dim': 4, 'hidden_dims': [64, 64], 'output_dim': 1, 'activation': 'ReLU'}, 'mlp_state_dict': model.state_dict(), 'input_normalization': {'feature_names': ['X1', 'X2', 'X3', 'X4'], 'mean': train_mean.detach().cpu().clone(), 'std': train_std.detach().cpu().clone(), 'std_unbiased': False}, 'generator_configs': {coordinate: {'num_bins': generator.num_bins, 'bound': generator.bound} for coordinate, generator in generators.items()}, 'generator_state_dicts': {coordinate: generator.state_dict() for coordinate, generator in generators.items()}, 'mlp_optimizer_state_dict': mlp_optimizer.state_dict(), 'generator_optimizer_state_dicts': {coordinate: optimizer.state_dict() for coordinate, optimizer in optimizers.items()}, 'multipliers': dict(multipliers), 'sampling_rng_device': str(rng.device), 'sampling_rng_state': rng.get_state(), 'torch_rng_state': torch.get_rng_state(), 'cuda_rng_states': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [], 'metrics': {'step': 0, 'id_validation_mse': initial_validation_mse, **initial_ood_metrics}, 'selection': {'metric': selection_key, 'mode': 'min', 'candidate_start_step': 0, 'tie_break': 'earliest', 'current_score': best_score, 'best_score': best_score, 'best_step': best_step}, 'torch_version': str(torch.__version__), 'numpy_version': str(np.__version__)}
        save_checkpoint(best_checkpoint_path, initial_checkpoint)
        save_checkpoint(last_checkpoint_path, initial_checkpoint)
        print(f'Saved initial ERM checkpoint at step 0: {selection_key}={best_score:.8f}')
    for step in range(start_step, args.steps + 1):
        print(f'\n=== Alternating step {step}/{args.steps} ===')
        generator_metrics = {}
        for coordinate in selected_coordinates:
            print(f'\n--- {coordinate}: {args.inner_steps} inner adversarial updates ---')
            metrics = None
            for inner_step in range(1, args.inner_steps + 1):
                print(f'\n[{coordinate}] inner step {inner_step}/{args.inner_steps}')
                noise = torch.randn(batch_size, 5, generator=rng, dtype=torch.float32).to(device)
                updated_multiplier, metrics = update_generator(generators=generators, model=model, selected_coordinate=coordinate, nominal_u=nominal_samples[coordinate], noise=noise, optimizer=optimizers[coordinate], multiplier=multipliers[coordinate], kl_budget=args.kl_budget, dual_lr=args.dual_lr, train_mean=train_mean, train_std=train_std)
                multipliers[coordinate] = updated_multiplier
            if metrics is None:
                raise RuntimeError(f'No generator update executed for {coordinate}.')
            generator_metrics[coordinate] = metrics
        mlp_noise = torch.randn(batch_size, 5, generator=rng, dtype=torch.float32).to(device)
        print('\nUpdating MLP:')
        mse_before, mse_after = update_mlp(generators=generators, model=model, noise=mlp_noise, optimizer=mlp_optimizer, train_mean=train_mean, train_std=train_std)
        history.append((step, mse_before, mse_after))
        validation_mse = evaluate_mse(model, validation_x, validation_y, train_mean, train_std)
        print(f'Step {step:04d} ID validation MSE: {validation_mse:.8f}')
        ood_metrics = evaluate_ood(model, ood_environments, train_mean, train_std)
        for strength in ['mild', 'moderate', 'strong']:
            key = f'ood_val_{strength}_mean_mse'
            print(f'OOD validation {strength} mean MSE: {ood_metrics[key]:.8f}')
        print(f"OOD validation mean MSE: {ood_metrics['ood_val_mean_mse']:.8f}")
        print(f"OOD validation worst MSE: {ood_metrics['ood_val_worst_mse']:.8f}")
        row = {'step': step, 'mlp_mse_before': mse_before, 'mlp_mse_after': mse_after, 'id_validation_mse': validation_mse}
        row.update(ood_metrics)
        for coordinate in selected_coordinates:
            metrics = generator_metrics[coordinate]
            row[f'{coordinate}_kl'] = metrics['kl']
            row[f'{coordinate}_kl_residual'] = metrics['kl_residual']
            row[f'{coordinate}_multiplier'] = multipliers[coordinate]
        with log_path.open('a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writerow(row)
        checkpoint = {'format_version': 1, 'method': 'ours', 'step': step, 'config': run_config, 'log_path': str(log_path.resolve()), 'mlp_architecture': {'input_dim': 4, 'hidden_dims': [64, 64], 'output_dim': 1, 'activation': 'ReLU'}, 'mlp_state_dict': model.state_dict(), 'input_normalization': {'feature_names': ['X1', 'X2', 'X3', 'X4'], 'mean': train_mean.detach().cpu().clone(), 'std': train_std.detach().cpu().clone(), 'std_unbiased': False}, 'generator_configs': {coordinate: {'num_bins': generator.num_bins, 'bound': generator.bound} for coordinate, generator in generators.items()}, 'generator_state_dicts': {coordinate: generator.state_dict() for coordinate, generator in generators.items()}, 'mlp_optimizer_state_dict': mlp_optimizer.state_dict(), 'generator_optimizer_state_dicts': {coordinate: optimizer.state_dict() for coordinate, optimizer in optimizers.items()}, 'multipliers': dict(multipliers), 'sampling_rng_device': str(rng.device), 'sampling_rng_state': rng.get_state(), 'torch_rng_state': torch.get_rng_state(), 'cuda_rng_states': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [], 'metrics': dict(row), 'torch_version': str(torch.__version__), 'numpy_version': str(np.__version__)}
        current_score = ood_metrics[selection_key]
        if not math.isfinite(current_score):
            raise ValueError('Checkpoint selection score is not finite.')
        improved = current_score < best_score
        if improved:
            best_score = current_score
            best_step = step
        checkpoint['selection'] = {'metric': selection_key, 'mode': 'min', 'candidate_start_step': 0, 'tie_break': 'earliest', 'current_score': current_score, 'best_score': best_score, 'best_step': best_step}
        if improved:
            save_checkpoint(best_checkpoint_path, checkpoint)
            print(f'Saved best checkpoint at step {step}: {selection_key}={best_score:.8f}')
        save_checkpoint(last_checkpoint_path, checkpoint)
        print(f'Saved last checkpoint at step {step}.')
        print(f'[PASS] Alternating step {step} completed.')
    print('\nMLP update summaries:')
    for step, mse_before, mse_after in history:
        print(f'Step {step:04d}: {mse_before:.8f} -> {mse_after:.8f}')
    print('\nFinal multipliers:')
    for coordinate in selected_coordinates:
        print(f'{coordinate}: {multipliers[coordinate]:.8f}')
    saved = torch.load(last_checkpoint_path, map_location='cpu', weights_only=True)
    if saved['step'] != args.steps:
        raise ValueError('Unexpected checkpoint step.')
    models_to_check = [('MLP', model, saved['mlp_state_dict'])]
    for coordinate, generator in generators.items():
        models_to_check.append((coordinate, generator, saved['generator_state_dicts'][coordinate]))
    for label, module, saved_state in models_to_check:
        for name, value in module.state_dict().items():
            if not torch.equal(value.detach().cpu(), saved_state[name]):
                raise ValueError(f'Checkpoint mismatch: {label}.{name}')
    if saved['multipliers'] != multipliers:
        raise ValueError('Checkpoint multiplier mismatch.')
    print(f'Last checkpoint: {last_checkpoint_path.resolve()}')
    print('[PASS] Saved model parameters and multipliers match.')
    saved_best = torch.load(best_checkpoint_path, map_location='cpu', weights_only=True)
    if saved_best['step'] != best_step:
        raise ValueError('Best checkpoint step mismatch.')
    if saved_best['selection']['metric'] != selection_key:
        raise ValueError('Best checkpoint selection metric mismatch.')
    if saved_best['metrics'][selection_key] != best_score:
        raise ValueError('Best checkpoint score mismatch.')
    best_model = copy.deepcopy(model)
    best_model.load_state_dict(saved_best['mlp_state_dict'])
    parameter = next(best_model.parameters())
    normalization = saved_best['input_normalization']
    best_mean = normalization['mean'].to(device=parameter.device, dtype=parameter.dtype)
    best_std = normalization['std'].to(device=parameter.device, dtype=parameter.dtype)
    restored_metrics = evaluate_ood(best_model, ood_environments, best_mean, best_std)
    restored_score = restored_metrics[selection_key]
    if not math.isclose(restored_score, best_score, rel_tol=1e-06, abs_tol=1e-07):
        raise ValueError('Restored best checkpoint score mismatch.')
    print(f'Best checkpoint: {best_checkpoint_path.resolve()}')
    print(f'Best step: {best_step}')
    print(f'Best validation score: {best_score:.8f}')
    print('[PASS] Best checkpoint reproduces its validation score.')
    print('[PASS] All alternating updates completed.')
if __name__ == '__main__':
    main()
