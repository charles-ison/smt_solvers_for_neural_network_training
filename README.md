# theory_of_computation_final

## Runing SMT vs Backpropagation Comparison ##
In order to run the comparison between the SMT trainer and the backpropagation trainer, please use either the code/run_training_comparison.ipynb Jupyter notebook or the code/run_training_comparison.py file (whichever you prefer)

## File Notes ##

 * code/model.py contains the simple feedforward neural network for testing
 * code/backpropagation_trainer.py file contains the code to run backpropagation training with PyTorch
 * code/smt_training.py contains the code to run the SMT training using pySMT and Z3
 * code/testing.py allows for testing either training method

## Virtual Environment Notes ##

How to start virtual environment and run jupyter notebook:

```source env/bin/activate ```

```python -m ipykernel install --user --name=my-env ```

```jupyter-notebook ```
