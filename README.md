https://mlsysbook.ai/tinytorch/intro.html

Setup uses tinytorch venv:
source ../tinytorch/.venv/bin/activate

# Graph visualization:

## Setup:
sudo apt update
sudo apt install graphviz

## Execution
- Default named nodes: python scripts/render_graph.py --example basic --output graph
- Show full tensor data: python scripts/render_graph.py --example basic --output graph --tensor-values
- Hide grad stats: python scripts/render_graph.py --example basic --output graph --no-grad-stats
- Full grads: python scripts/render_graph.py --example basic --output graph --grad-values