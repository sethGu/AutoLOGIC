"""Generate the redistributable synthetic binary-classification example."""
from pathlib import Path
import csv
import random


def generate(destination: Path) -> None:
    rng = random.Random(42)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'target'])
        for _ in range(240):
            x = [round(rng.gauss(0, 1), 6) for _ in range(6)]
            score = 1.4*x[0] - x[1] + 0.7*x[2]*x[3] + 0.15*rng.gauss(0, 1)
            writer.writerow(x + [int(score > 0)])


if __name__ == '__main__':
    generate(Path(__file__).parent / 'data' / 'demo_binary.csv')
