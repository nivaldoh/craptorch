https://mlsysbook.ai/tinytorch/intro.html

Setup uses tinytorch venv:
source ../tinytorch/.venv/bin/activate

# Graph visualization:

## Setup:
sudo apt update
sudo apt install graphviz

## Execution
- Default named nodes: python scripts/render_graph.py --example basic --output mnt/c/users/niv/desktop/graph
- Show full tensor data: python scripts/render_graph.py --example basic --output graph --tensor-values
- Hide grad stats: python scripts/render_graph.py --example basic --output graph --no-grad-stats
- Full grads: python scripts/render_graph.py --example basic --output graph --grad-values

# Matmul visualization:

## Execution:
- Default example: python scripts/visualize_matmul.py --output mnt/c/users/niv/desktop/matmul.png
- Custom matrices: python scripts/visualize_matmul.py --a "1 2; 3 4" --b "5 6; 7 8" --cell 1,0 --output mnt/c/users/niv/desktop/matmul.png
- Random matrices: python scripts/visualize_matmul.py --a-shape 2x3 --b-shape 3x2 --seed 7 --output mnt/c/users/niv/desktop/matmul.png
