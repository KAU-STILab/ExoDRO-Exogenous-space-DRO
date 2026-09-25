import copy
import argparse
import math
from pathlib import Path

import numpy as np
import torch

from exogenous_generator import ExogenousGenerator


def load_u(path, coordinate):
    """Read the selected exogenous coordinate."""
    data = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=np.float64,
        encoding="utf-8",
    )

    if data.dtype.names is None or coordinate not in data.dtype.names:
        raise ValueError(f"Missing {coordinate} column: {path}")

    values = np.atleast_1d(data[coordinate]).copy()

    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"Empty or non-finite {coordinate} data: {path}")

    """Convert CSV float64 values to the generator's float32 dtype."""
    return torch.tensor(values, dtype=torch.float32)


def evaluate_nll(generator, values):
    """Evaluate without computing gradients."""
    generator.clear_cache()

    with torch.no_grad():
        nll = -generator.log_prob(values).mean()

    generator.clear_cache()

    if not torch.isfinite(nll).item():
        raise ValueError("Evaluation NLL is not finite.")

    return nll.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation_id", type=Path, required=True)
    parser.add_argument(
        "--coordinate",
        choices=["U1", "U2", "U3", "U4"],
        default="U1",
        help="Exogenous coordinate to fit.",
    )

    parser.add_argument("--bound", type=float, default=3.0)
    parser.add_argument(
    "--save-dir",
    type=Path,
    default=Path(__file__).resolve().parent / "checkpoints",
    )

    args = parser.parse_args()

    torch.manual_seed(0)

    """Development settings, not finalized paper settings."""
    steps = 3000
    learning_rate = 0.001

    train_u = load_u(args.train, args.coordinate)
    validation_u = load_u(args.validation_id, args.coordinate)

    generator = ExogenousGenerator(num_bins=8, bound=args.bound)

    optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=learning_rate,
    )

    initial_train_nll = evaluate_nll(generator, train_u)
    initial_validation_nll = evaluate_nll(generator, validation_u)

    print(f"Train samples: {len(train_u)}")
    print(f"Validation samples: {len(validation_u)}")
    print(f"Initial train NLL: {initial_train_nll:.6f}")
    print(f"Initial validation NLL: {initial_validation_nll:.6f}")

    """Reference NLL of the known nominal distribution N(0, 1)."""
    reference_nll = (
        0.5 * validation_u.square()
        + 0.5 * math.log(2.0 * math.pi)
    ).mean().item()

    print(f"Standard normal validation NLL: {reference_nll:.6f}")

    """Keep the initial model as the first candidate."""
    best_validation_nll = initial_validation_nll
    best_step = 0
    best_state = copy.deepcopy(generator.state_dict())

    for step in range(1, steps + 1):
        generator.clear_cache()
        optimizer.zero_grad(set_to_none=True)

        """Fit the generator to the fixed nominal training samples."""
        loss = -generator.log_prob(train_u).mean()

        if not torch.isfinite(loss).item():
            raise ValueError(f"Non-finite loss at step {step}")

        loss.backward()

        for name, parameter in generator.named_parameters():
            if parameter.grad is None:
                raise ValueError(f"Missing gradient: {name}")

            if not torch.isfinite(parameter.grad).all().item():
                raise ValueError(f"Non-finite gradient: {name}")

        optimizer.step()

        """Cached transformations are stale after parameter updates."""
        generator.clear_cache()

        """Evaluate after every parameter update."""
        validation_nll = evaluate_nll(generator, validation_u)

        """Keep a copy whenever validation NLL improves."""
        if validation_nll < best_validation_nll:
            best_validation_nll = validation_nll
            best_step = step
            best_state = copy.deepcopy(generator.state_dict())

        """Print progress every 100 steps."""
        if step == 1 or step % 100 == 0:
            train_nll = evaluate_nll(generator, train_u)

            print(
                f"Step {step:4d} | "
                f"Train NLL: {train_nll:.6f} | "
                f"Validation NLL: {validation_nll:.6f} | "
                f"Best step: {best_step}"
            )

    final_validation_nll = evaluate_nll(
        generator, validation_u
    )

    print("\nTraining finished.")
    print(
        "Validation NLL change: "
        f"{initial_validation_nll:.6f} "
        f"-> {final_validation_nll:.6f}"
    )

    """Restore the model with the lowest validation NLL."""
    generator.load_state_dict(best_state)
    generator.clear_cache()

    nll_before_save = evaluate_nll(generator, validation_u)

    """Create the checkpoint directory."""
    save_dir = args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = save_dir / f"nominal_{args.coordinate.lower()}.pt"

    """Store model parameters and training settings."""
    checkpoint = {
        "coordinate": args.coordinate,
        "num_bins": generator.num_bins,
        "bound": generator.bound,
        "seed": 0,
        "learning_rate": learning_rate,
        "total_steps": steps,
        "best_step": best_step,
        "best_validation_nll": best_validation_nll,
        "train_path": str(args.train.resolve()),
        "validation_path": str(args.validation_id.resolve()),
        "state_dict": best_state,
    }

    torch.save(checkpoint, checkpoint_path)

    print(f"\nBest step: {best_step}")
    print(f"Best validation NLL: {nll_before_save}")
    print(f"Saved to: {checkpoint_path}")


    "Read the saved parameters and settings."
    loaded = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    "Create a new generator with the saved settings."
    restored_generator = ExogenousGenerator(
        num_bins=loaded["num_bins"],
        bound=loaded["bound"],
    )

    "Restore the saved parameters."
    restored_generator.load_state_dict(loaded["state_dict"])
    restored_generator.clear_cache()

    "Evaluate the restored generator on the same validation data."
    nll_after_load = evaluate_nll(
        restored_generator, validation_u
    )

    "Check that saving and loading preserved the result."
    if not math.isclose(
        nll_before_save,
        nll_after_load,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise ValueError("Validation NLL changed after loading.")

    print(f"Validation NLL before saving: {nll_before_save}")
    print(f"Validation NLL after loading: {nll_after_load}")
    print("[PASS] Checkpoint reload verification")


if __name__ == "__main__":
    main()