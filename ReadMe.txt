This project contains all the source codes.

There are 6 packages and 1 utils in this projects.
  |
  |------ numpynet -- My CNN deep learning layers implemented by numpy.
  |                |
  |                | ---- layers.py
  |                | ---- loss.py
  |                | ---- optimizer.py
  |
  |------ numpy_models -- Network model constructed by my numpy layers.
  |                |
  |                | ---- models.py
  |
  |------ pytorch_models -- Network model constructed by pytorch.
  |                |
  |                | ---- models.py
  |
  |------ sudokusolver
  |                |
  |                | ---- DancingList.py
  |                | ---- sudoku_solver.py -- This file contains 4 puzzle examples in different difficulties and show how to use my sudoku solver.
  |
  |------ dataset_wrapper
  |                |
  |                | ---- dataset_wrapper.py -- This file provides a convient wrapper for training. It provides three dataset, MNIST, CN and Full. It splits data automatically.
  |
  |------ pyimagesearch -- Provided by OpenCV and I modified puzzle.py
  |
  |------ utils.py -- I provided a result recorder in this file with some other tools.
  |

There are 4 executable python files.
  |
  |------ train_numpynet.py
  |
  |------ train_pytorch.py
  |
  |------ iris_test.py -- a simple test file with iris dataset.
  |
  |------ solve_sudoku_puzzle.py -- This file is main file to solve problem. Usage is same as OpenCV's standard example.
  |

Datasets are all stored in diractory -- datasets. All data used for training are from here.

Trained model are save in outputs. I provided a well trained pytorch model for project sudoku puzzle and a LeNet-5 model for MINST implemented by my numpy net.

Test cases provided by teaching assistants are all in -- test1

*Some explanations for special file postfix.
.pkl files are my trained numpy net model.
.dic files are the state dictionaries of pytorch model and its optimizer.
.rec files are records generate by the result recorder I provided. Use to plot the result.
If you only save the state_dict of network, you should modify the model loading part in solve_sudoku_puzzle.py.




